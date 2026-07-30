"""Two-stage transfer-learning trainer.

The experiment compares two splits, so this script must behave identically regardless
of which one it is given. `--split` selects the data; everything else comes from
config.yaml. Nothing about the model, schedule, augmentation or seed varies.

Stage 1 trains the head on frozen features. Stage 2 unfreezes and fine-tunes at a
lower learning rate. Model selection uses **balanced accuracy on validation**, not
accuracy -- with 67% `nv`, plain accuracy would happily select a model that never
predicts melanoma.

Usage
-----
    python -m src.models.train --split grouped
    python -m src.models.train --split naive
    python -m src.models.train --split grouped --smoke-test   # 200 imgs, 1 epoch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, set_seed
from src.models.build import build_model, resolve_device, set_backbone_trainable
from src.models.dataset import (
    assert_no_group_leakage,
    compute_class_weights,
    load_split,
    make_loader,
)

SPLIT_KEYS = {"naive": "splits.naive_name", "grouped": "splits.grouped_name"}


@dataclass
class EpochResult:
    epoch: int
    stage: str
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_balanced_accuracy: float
    seconds: float


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall. Classes absent from y_true are skipped, not counted as 0."""
    recalls = []
    for cls in range(n_classes):
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def evaluate_epoch(model, loader, criterion, device, n_classes: int):
    """Run one validation pass. Returns (loss, accuracy, balanced_accuracy)."""
    import torch

    model.eval()
    total_loss = 0.0
    n_seen = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += float(loss.item()) * images.size(0)
            n_seen += images.size(0)
            all_true.append(targets.cpu().numpy())
            all_pred.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return (
        total_loss / max(n_seen, 1),
        float((y_true == y_pred).mean()),
        balanced_accuracy(y_true, y_pred, n_classes),
    )


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None) -> float:
    import torch

    model.train()
    total_loss = 0.0
    n_seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        n_seen += images.size(0)

    return total_loss / max(n_seen, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train on one split.")
    parser.add_argument("--split", required=True, choices=sorted(SPLIT_KEYS),
                        help="which split to train on (this is the experiment's variable)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="tiny subset, 1 epoch per stage — verifies the pipeline runs")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="override config; use 0 if DataLoader workers misbehave")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    import torch

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    torch.use_deterministic_algorithms(False)   # cuDNN determinism costs too much here

    classes = cfg.classes
    n_classes = len(classes)
    image_dir = cfg.path("paths.raw_dir") / "images"
    splits_dir = cfg.path("paths.splits_dir")
    split_file = cfg.get(SPLIT_KEYS[args.split])

    # --- data ---------------------------------------------------------------------
    full = load_split(splits_dir, split_file)
    if args.split == "grouped":
        # The honest split must be clean. If it isn't, every downstream number is void.
        assert_no_group_leakage(full)
        print("✓ Grouped split verified clean (no group spans partitions)")
    else:
        spans = full.groupby("group")["split"].nunique()
        print(f"⚠ Naive split: {int((spans > 1).sum()):,} group(s) span partitions "
              f"— this is the deliberately leaky control")

    train_df = full[full["split"] == "train"].reset_index(drop=True)
    val_df = full[full["split"] == "val"].reset_index(drop=True)

    if args.smoke_test:
        train_df = train_df.groupby("dx", group_keys=False).head(20).reset_index(drop=True)
        val_df = val_df.groupby("dx", group_keys=False).head(10).reset_index(drop=True)
        print(f"SMOKE TEST: {len(train_df)} train / {len(val_df)} val images")

    strategy = cfg.get("train.imbalance_strategy", "class_weights")
    batch_size = int(cfg.get("train.batch_size", 32))
    image_size = int(cfg.get("model.image_size", 224))
    if args.num_workers is not None:
        num_workers = args.num_workers
    elif args.smoke_test:
        num_workers = 0
    else:
        num_workers = int(cfg.get("train.num_workers", 4))

    train_loader = make_loader(train_df, image_dir, classes, image_size, batch_size,
                               train=True, imbalance_strategy=strategy,
                               num_workers=num_workers, seed=cfg.seed)
    val_loader = make_loader(val_df, image_dir, classes, image_size, batch_size,
                             train=False, num_workers=num_workers, seed=cfg.seed)

    # --- model --------------------------------------------------------------------
    device = resolve_device()
    model = build_model(
        backbone=cfg.get("model.backbone", "efficientnet_b0"),
        n_classes=n_classes,
        pretrained=bool(cfg.get("model.pretrained", True)),
        dropout=float(cfg.get("model.dropout", 0.2)),
    ).to(device)

    criterion_weight = None
    if strategy == "class_weights":
        weights = compute_class_weights(train_df["dx"], classes)
        criterion_weight = torch.as_tensor(weights, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=criterion_weight)

    use_amp = bool(cfg.get("train.mixed_precision", True)) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"\nSplit      : {args.split} ({split_file})")
    print(f"Device     : {device}")
    print(f"Backbone   : {cfg.get('model.backbone')}")
    print(f"Imbalance  : {strategy}")
    print(f"Train/Val  : {len(train_df):,} / {len(val_df):,}")

    # --- training -----------------------------------------------------------------
    epochs_head = 1 if args.smoke_test else int(cfg.get("train.epochs_head", 3))
    epochs_ft = 1 if args.smoke_test else int(cfg.get("train.epochs_finetune", 12))
    patience = int(cfg.get("train.early_stopping_patience", 4))

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{args.split}_best.pt"

    history: list[EpochResult] = []
    best_metric = -np.inf
    epochs_without_improvement = 0
    epoch_counter = 0

    for stage, n_epochs, lr, trainable in (
        ("head", epochs_head, float(cfg.get("train.lr_head", 1e-3)), False),
        ("finetune", epochs_ft, float(cfg.get("train.lr_finetune", 1e-4)), True),
    ):
        n_trainable, n_total = set_backbone_trainable(model, trainable)
        print(f"\n--- stage: {stage} | lr={lr} | trainable {n_trainable:,}/{n_total:,} ---")

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=float(cfg.get("train.weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))

        for _ in range(n_epochs):
            epoch_counter += 1
            started = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_acc, val_bal = evaluate_epoch(model, val_loader, criterion,
                                                        device, n_classes)
            scheduler.step()

            result = EpochResult(epoch_counter, stage, train_loss, val_loss,
                                 val_acc, val_bal, time.time() - started)
            history.append(result)
            print(f"  epoch {epoch_counter:>2} | train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| acc {val_acc:.4f} | bal_acc {val_bal:.4f} | {result.seconds:.0f}s")

            if val_bal > best_metric:
                best_metric = val_bal
                epochs_without_improvement = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "split": args.split,
                    "backbone": cfg.get("model.backbone"),
                    "classes": classes,
                    "epoch": epoch_counter,
                    "val_balanced_accuracy": val_bal,
                }, checkpoint_path)
                print(f"       ↳ new best (bal_acc {val_bal:.4f}) — saved")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"  early stopping after {patience} epochs without improvement")
                    break
        else:
            continue
        break

    # --- record -------------------------------------------------------------------
    reports_dir = cfg.path("paths.reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(h) for h in history]).to_csv(
        reports_dir / f"training_history_{args.split}.csv", index=False
    )
    with open(reports_dir / f"training_meta_{args.split}.json", "w") as fh:
        json.dump({
            "split": args.split,
            "split_file": split_file,
            "backbone": cfg.get("model.backbone"),
            "imbalance_strategy": strategy,
            "seed": cfg.seed,
            "device": device,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "best_val_balanced_accuracy": best_metric,
            "epochs_run": epoch_counter,
            "smoke_test": args.smoke_test,
        }, fh, indent=2)

    print(f"\n✓ Best val balanced accuracy: {best_metric:.4f}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  History:    {reports_dir / f'training_history_{args.split}.csv'}")
    print(f"\nNext: python -m src.evaluate --split {args.split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
