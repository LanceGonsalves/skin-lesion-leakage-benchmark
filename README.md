<h1 align="center">🔬 Skin Lesion Classification — A Leakage-Aware Benchmark</h1>

<p align="center">
  <i>Most skin-lesion classifiers are evaluated on splits that leak.<br>
  This project measures how much that inflates results — and re-benchmarks honestly.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Dataset-HAM10000-4C6EF5?style=flat-square">
  <img src="https://img.shields.io/badge/Domain-Medical%20Imaging-8E44AD?style=flat-square">
  <img src="https://img.shields.io/badge/status-work%20in%20progress-F39C12?style=flat-square">
</p>

> ⚠️ **Not a clinical tool.** This is a research and educational project. It is not validated
> for diagnostic use and no clinical claims are made.

---

## The problem

[HAM10000](https://www.nature.com/articles/sdata2018161) is one of the most widely used
dermatoscopic image datasets — 10,015 images across 7 diagnostic classes. Portfolio projects
and papers alike routinely report 90%+ accuracy on it.

There's a catch buried in the metadata. Each row has **both** an `image_id` and a `lesion_id`,
and **multiple images map to the same physical lesion** — the same mole photographed more than
once, at different angles or magnifications.

Split the data randomly *by image* — which is what most tutorials do — and images of the **same
lesion land in both the training and test sets**. The model no longer has to learn what melanoma
looks like; it can recognise *that particular mole*. Test accuracy goes up. The model gets worse.

Independent audits have also found **true duplicate pairs beyond the metadata**: image pairs
labelled as different lesions that manual review confirms are the same lesion, some of which
straddle the train/test boundary.

## The question

> **How much of the reported performance of skin-lesion classifiers is an artefact of data
> leakage — and what does an honest benchmark look like?**

## The approach

The experiment is deliberately simple, because the *split* is the independent variable:

1. **Audit** the dataset for duplicate and near-duplicate images — via perceptual hashing and
   CNN embedding similarity — on top of the lesion grouping already visible in the metadata.
2. **Build two splits** with identical ratios, stratification and seed:
   - **Split A (naive)** — random at the *image* level. Leaky by construction. Kept on purpose.
   - **Split B (grouped)** — grouped by `lesion_id`, with discovered duplicate groups merged in.
3. **Train the same model twice**, changing nothing but the split.
4. **The difference in test performance is the leakage effect.** That gap is the result.
5. **Benchmark honestly** on Split B with the metrics imbalanced medical data actually requires —
   balanced accuracy, macro-F1, per-class recall, bootstrap confidence intervals, calibration.
6. **Interpret** with Grad-CAM, checking whether the model attends to the lesion or to imaging
   artefacts (rulers, ink marks, hair, vignetting).

This project is **not** attempting state-of-the-art. It's attempting an honest number.

## Results

> 🚧 *Experiments in progress — this section will carry the headline comparison table.*

| | Split A (naive) | Split B (grouped) | Δ |
|---|---|---|---|
| Balanced accuracy | — | — | — |
| Macro F1 | — | — | — |
| Melanoma recall | — | — | — |

## Dataset

**HAM10000** — Tschandl, Rosendahl & Kittler, *Scientific Data* (2018).
10,015 dermatoscopic images; classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.

The images are **CC BY-NC licensed and not redistributed here.** Download them yourself:

- **Harvard Dataverse** — [DOI 10.7910/DVN/DBW86T](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- **Kaggle** — `kmader/skin-cancer-mnist-ham10000` (convenient if you want free GPU alongside)

Place `HAM10000_metadata.csv` and the image files under `data/raw/`.

## Quickstart

```bash
pip install -r requirements.txt
python -m src.data.download --check     # verify the dataset is present and intact
```

More commands land as the pipeline is built out.

## Repo layout

```
skin-lesion-leakage-benchmark/
├── config.yaml            # single source of truth for every experiment
├── src/
│   ├── config.py          # config loading + seeding
│   ├── data/              # download, audit, splits
│   └── models/            # build, train
├── notebooks/             # the narrative: audit → experiment → interpretability
├── reports/figures/       # confusion matrices, Grad-CAM grids, duplicate contact sheets
└── tests/                 # incl. a test asserting no lesion_id spans two splits
```

## Limitations

- HAM10000's demographics skew towards lighter skin types; results should not be assumed to
  generalise across skin tones. See the published work on
  [racial bias in dermoscopy repositories](https://onlinelibrary.wiley.com/doi/full/10.1002/jvc2.477).
- Automated duplicate detection produces false positives; flagged pairs are manually verified on
  a sample and detector precision is reported rather than assumed.
- Single train/val/test split rather than full cross-validation, for compute reasons.

## References

1. Tschandl, Rosendahl & Kittler (2018). *The HAM10000 dataset.* Scientific Data. [link](https://www.nature.com/articles/sdata2018161)
2. Cassidy et al. (2022). *Analysis of the ISIC image datasets: usage, benchmarks and recommendations.* Medical Image Analysis.
3. *Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological Image Datasets* (2025). Scientific Data. [link](https://www.nature.com/articles/s41597-025-04382-5)

## Author

**Lance Gonsalves** · [GitHub](https://github.com/LanceGonsalves) · [LinkedIn](https://www.linkedin.com/in/lance-gonsalves/)
