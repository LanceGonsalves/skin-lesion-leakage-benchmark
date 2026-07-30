"""Model construction.

Deliberately minimal. The experiment varies the *split*, so the architecture must be
byte-identical between runs — every knob lives in config.yaml and nothing is decided
at call time.
"""

from __future__ import annotations


def resolve_device(prefer: str | None = None) -> str:
    """Pick the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    import torch

    if prefer and prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(backbone: str, n_classes: int, pretrained: bool = True,
                dropout: float = 0.2):
    """Create a timm backbone with a fresh classification head.

    `num_classes` makes timm swap in a correctly-sized head automatically, so the
    pretrained ImageNet classifier is discarded rather than adapted.
    """
    import timm

    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=n_classes,
        drop_rate=dropout,
    )


def set_backbone_trainable(model, trainable: bool) -> tuple[int, int]:
    """Freeze or unfreeze everything except the classification head.

    Stage 1 trains only the head on frozen features -- fast, and it stops a randomly
    initialised head from destroying pretrained weights with large early gradients.
    Stage 2 unfreezes for fine-tuning at a lower learning rate.

    Returns (trainable_params, total_params).
    """
    import timm

    classifier = timm.models.get_classifier(model) if hasattr(timm.models, "get_classifier") \
        else model.get_classifier()
    head_params = {id(p) for p in classifier.parameters()}

    for param in model.parameters():
        param.requires_grad = trainable if id(param) not in head_params else True

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable_count, total
