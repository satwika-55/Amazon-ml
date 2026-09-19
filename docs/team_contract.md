# Team Contract

## Project
Amazon ML Challenge 2026 multi-modal prediction pipeline.

## Team Roles

| Member | Area | Primary responsibility |
| --- | --- | --- |
| Member 1 | Data + Tabular | Canonical processed data, validation, tabular feature engineering, and tabular baselines |
| Member 2 | Text / NLP | Text cleaning, TF-IDF, transformer embeddings, and text model predictions |
| Member 3 | Image / Vision | Image validation, preprocessing, frozen embeddings, and image model predictions |
| Member 4 | Fusion + Ensemble | Prediction alignment, validation comparison, blending, stacking, and final submission |

## Core Working Rules

1. Use the shared raw data location only as input. Do not modify files in `data/raw/`.
2. All modality-specific teams must consume the shared canonical processed schema agreed by the team.
3. Each model output must follow the task-specific prediction contract:
	- Regression and binary classification: `id,prediction`.
	- Multiclass classification: `id,prob_<class_0>,prob_<class_1>,...`, using the exact order configured in `task.classes`.
4. Validation performance is the decision criterion for keeping or discarding a model.
5. More models are only kept when they add complementary information and improve validation metrics.
6. No unnecessary framework or infrastructure work should be introduced before the modeling milestones are complete.

## Communication Standard

- Announce interface changes before merging them into shared code paths.
- Document assumptions and known quality issues in the relevant milestone notes.
- Keep feature engineering and preprocessing logic reproducible and versioned.
- Confirm artifact and model paths before running final submission pipelines.

## Merge and Quality Expectations

Before merging work into the main branch:

- Pull the latest upstream state.
- Run only the relevant checks for the modified component.
- Confirm no accidental data, checkpoints, or large model files are staged.
- Record metric, configuration, and artifact details for every experiment.
- Keep commit messages readable and specific.

## Success Criteria

The team succeeds when:

- data is normalized and shared across all modalities,
- each modality has at least one validated baseline model,
- fusion models improve or reliably match the best single-modality baseline,
- the final submission is generated from a documented pipeline with reproducible configuration.
