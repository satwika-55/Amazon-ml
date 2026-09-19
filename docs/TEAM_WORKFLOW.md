# Team Workflow

## Documentation Index

This directory contains the core planning and milestone documents for the project.

- [team_contract.md](team_contract.md): team roles, communication rules, and quality expectations
- [m1_spec.md](m1_spec.md): data contract and baseline setup
- [m2_spec.md](m2_spec.md): text and image model baselines
- [m3_spec.md](m3_spec.md): tabular feature engineering and stronger models
- [m4_fusion_spec.md](m4_fusion_spec.md): fusion, ensemble, and final submission

## Operating Rhythm

1. Complete the milestone requirements in the assigned area.
2. Keep validation outputs and config notes reproducible.
3. Share schema or interface changes before merging.
4. Only advance to the next milestone after recorded validation evidence.
5. Use the fusion stage only when the best model or ensemble is clearly justified.

## Delivery Standard

Every team member should ensure that:

- raw data is left unchanged,
- processed artifacts are stored in documented locations,
- model outputs use the common prediction schema,
- metrics and training notes are retained for decision-making,
- final submission work is based on the validated best pipeline.
🔥 Shared team contract — give this to everyone

I would put this in:

docs/team_contract.md
TEAM CONTRACT — MULTIMODAL ML COMPETITION
==========================================

1. ONE CONFIG
--------------

config/config.yaml

No separate M1/M2/M3/M4 configs.

Everything competition-specific must be configurable.


2. SHARED TASK

task.type
task.target
task.id_column
task.classes


3. SHARED METRIC

Everyone imports:

src/utils/metrics.py

Never implement separate metric formulas.

Metric direction is explicit.


4. SHARED FOLDS

folds/folds.csv

Schema:

id,fold

Everyone uses the same folds.

No independent fold generation.


5. ID

IDs are strings.

Preserve:

- leading zeros
- case
- normalized whitespace

Never merge by row position.


6. OOF CONTRACT

predictions/oof/<model>_oof.csv

Regression:

id,prediction

Binary:

id,prediction

Multiclass:

id,prob_class_0,...

Every training ID exactly once.


7. TEST CONTRACT

predictions/test/<model>_test.csv

Every test ID exactly once.


8. ORIGINAL TARGET SCALE

Every prediction file is written in ORIGINAL target scale.

If model internally uses:

log1p(y)

inverse-transform before writing predictions.


9. NO NaN

Base models must produce valid predictions for every ID.

Missing modality ≠ missing prediction.

M4 never fills NaN.


10. TEXT STRUCTURED FEATURES

M2 may produce:

features/text_struct.csv

Example:

id,brand,pack_quantity,weight,unit,...

These are label-free unless explicitly documented.

Target-derived features MUST be fold-fitted.


11. MISSING IMAGE

M3 must provide a fallback prediction.


12. OOF LEAKAGE

OOF prediction for row i must come from a model that did not train on row i.


13. M4

M4 consumes predictions.

M4 does not retrain M1/M2/M3 internals.


14. SAMPLE SUBMISSION

Final submission must follow:

data.sample_submission_path

Row order follows sample submission.


15. EXPERIMENTS

Every meaningful experiment records:

- what changed
- why
- metric
- score
- compute cost
- decision


16. COMPUTE

Cheap → strong.

Do not waste competition time on expensive models without evidence.


17. ADAPTABILITY

Nothing should assume:

- House Prices
- RMSE
- SMAPE
- exactly 3 models
- exactly 3 modalities
- a specific number of folds

The competition config decides.


18. FINAL RULE

Evidence beats assumptions.

A complex model that does not improve validation is not automatically better.

A modality that does not improve fusion does not have to be used.

The best single model is always a valid final solution.