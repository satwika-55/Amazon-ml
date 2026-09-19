# Amazon ML Challenge 2026

## Project Objective

Build a multimodal machine learning pipeline that combines:

- Text
- Images
- Tabular and structured data
- Domain-specific feature engineering
- Multiple candidate models
- Fusion, ensembling, and stacking
- Final prediction and submission generation

The team will evaluate each component with validation data and combine models only when they provide useful, complementary errors. More models do not automatically mean better performance.

## Pipeline Architecture

```text
RAW DATA
    ↓
DATA CLEANING / PREPROCESSING
    ↓
 ┌──────────────┬──────────────┬──────────────┐
 │     TEXT     │    IMAGE     │   TABULAR    │
 └──────────────┴──────────────┴──────────────┘
          ↓
   MULTIPLE MODELS
          ↓
   OOF PREDICTIONS
          ↓
 FUSION / ENSEMBLE / STACKING
          ↓
   FINAL PREDICTION
          ↓
      SUBMISSION
```

## Team Responsibilities

| Member | Area | Primary responsibility |
| --- | --- | --- |
| Member 1 | Data + Tabular | Canonical processed data, validation, features, and tabular models |
| Member 2 | Text / NLP | Text preprocessing, TF-IDF, transformer embeddings, and text predictions |
| Member 3 | Image / Vision | Image validation, preprocessing, frozen image embeddings, and image predictions |
| Member 4 | Fusion + Ensemble | Prediction alignment, comparison, blending, stacking, and submission generation |

## Repository Structure

```text
amazon-ml-challenge-2026/
├── data/                 # Raw, processed, and external data locations
├── src/
│   ├── data/             # Loading, validation, and shared feature engineering
│   ├── text/             # Text preprocessing and text models
│   ├── image/            # Image preprocessing and image models
│   ├── tabular/          # Tabular preprocessing and tabular models
│   ├── fusion/           # Ensembling, stacking, and final model orchestration
│   └── utils/            # Metrics, CV, logging, and configuration helpers
├── models/               # Model artifacts; large files stay out of Git
├── predictions/          # Intermediate prediction artifacts
├── submissions/          # Submission files
├── notebooks/            # Exploratory notebooks
├── docs/                 # Team process documentation
└── scripts/              # Explicit pipeline entry points
```

## How We Work

1. Member 1 creates and documents the canonical cleaned/processed dataset.
2. Members 2 and 3 consume the agreed input format instead of independently rewriting the raw dataset.
3. Members 1, 2, and 3 produce documented prediction files using the common schema `id,prediction`.
4. Member 4 aligns the prediction files, compares validation behavior, and evaluates blending or stacking.
5. Every experiment records its validation score, configuration, and relevant artifact locations.

Members should work primarily in their assigned folders and communicate interface changes before merging them.

## Git Workflow

```text
main
  |
  ├── member1-data-tabular
  ├── member2-text
  ├── member3-image
  └── member4-fusion
```

Before merging:

- Pull or rebase from `main`.
- Run the relevant checks for the changed area.
- Confirm that no accidental datasets, model weights, checkpoints, logs, or generated artifacts are staged.
- Use meaningful commit messages.

## Important Data Rules

- `data/raw/` is read-only.
- Never edit the original dataset.
- Never regenerate benchmark or validation data.
- Store processed datasets in `data/processed/`.
- Do not commit large datasets unless explicitly required.
- Do not commit model weights unless explicitly required.
- Predictions must follow a documented schema, initially `id,prediction`.
- Record every experiment's validation score and configuration.

## Important Project Rules

- Do not assume that adding more models improves performance.
- Add models based on validation performance and complementary errors.
- Do not add Docker, Kubernetes, MLflow, databases, cloud infrastructure, CI/CD, or unnecessary frameworks at this stage.
- Dependencies will be selected and installed later by the team; this repository setup does not install packages or download assets.