# Team Workflow

## Ownership

### Member 1: Data + Tabular

- Data loading
- Data validation
- Preprocessing
- Domain-specific feature extraction
- Tabular baseline
- Tabular advanced models

Member 1 owns the canonical cleaned/processed dataset. The input format should be agreed with the team before downstream modeling begins.

### Member 2: Text / NLP

- Text preprocessing
- TF-IDF baseline
- Transformer embedding extraction
- Text models
- Text predictions

Member 2 consumes the canonical agreed input format and produces documented text prediction artifacts.

### Member 3: Image / Vision

- Image preprocessing
- Image validation
- CLIP or other frozen image embeddings
- Image models
- Image predictions

Member 3 consumes the canonical agreed input format and produces documented image prediction artifacts.

### Member 4: Fusion + Ensemble

- Collect predictions from Members 1, 2, and 3
- Compare models
- Ensemble
- Weighted blending
- Stacking
- Final prediction
- Submission generation

Member 4 consumes prediction files produced by the other members. Prediction rows must be aligned by the shared identifier before evaluation or combination.

## Shared Interfaces

The initial common prediction format is:

```text
id,prediction
```

The actual dataset column names must be inspected and agreed later. Do not hardcode assumed dataset columns into this documentation or the placeholder modules.

Each prediction artifact should document its source model, split, feature version, and validation configuration alongside the file or in experiment notes.

## Data Rules

- `data/raw/` is read-only.
- Never edit the original dataset.
- Never regenerate benchmark or validation data.
- Processed datasets go into `data/processed/`.
- Large datasets should not be committed to Git unless explicitly required.
- Model weights should not be committed unless explicitly required.
- Predictions should follow the documented schema.
- Every experiment should record its validation score and configuration.

Members 2 and 3 must consume the agreed processed input rather than independently creating incompatible versions of the raw dataset.

## Branch Workflow

```text
main
  |
  ├── member1-data-tabular
  ├── member2-text
  ├── member3-image
  └── member4-fusion
```

Members should work primarily in their assigned folders. Before merging:

1. Pull or rebase from `main`.
2. Run relevant checks for the changed area.
3. Ensure no accidental dataset or model files are committed.
4. Use a meaningful commit message.

Changes to shared interfaces, processed-data schemas, or prediction schemas should be communicated to the whole team before merging.