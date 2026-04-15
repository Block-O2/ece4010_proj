# Additional Analysis

## Subgroup Descriptive Analysis by Animal Type

The processed dataset was compared across dogs and cats to show how the task differs across major subgroups.

| animal_type | rows | fast_adoption_rate | adoption_rate | median_age_days |
| --- | --- | --- | --- | --- |
| Cat | 69324 | 0.2783 | 0.5109 | 60.0 |
| Dog | 94608 | 0.3668 | 0.4982 | 365.0 |

## Random Forest Performance by Animal Type

The saved main random-forest model was applied to the deterministic stratified test split, and performance was computed separately for dogs and cats.

| animal_type | accuracy | f1_score | support | positive_rate | roc_auc |
| --- | --- | --- | --- | --- | --- |
| Cat | 0.7829 | 0.6497 | 13906 | 0.2774 | 0.8657 |
| Dog | 0.6892 | 0.6358 | 18881 | 0.3677 | 0.7712 |

## Intake-Year Ablation

A lightweight ablation removed `intake_year` from the main random-forest feature set and retrained only that model on the existing processed dataset.

Baseline RF metrics: Accuracy `0.7289`, F1-score `0.6406`, ROC-AUC `0.8149`.

Ablation RF metrics without `intake_year`: Accuracy `0.7151`, F1-score `0.6283`, ROC-AUC `0.8014`.

Change vs baseline: Accuracy `-0.0138`, F1-score `-0.0123`, ROC-AUC `-0.0135`.

Interpretation:

- If the drop is small, the model is not relying too heavily on year alone.
- If the drop is noticeable but not catastrophic, `intake_year` is useful but not the only driver.
- This ablation is intentionally lightweight and is meant to strengthen the final report rather than replace the main baseline.

## Rerun Notes

- The full main evaluation was not rerun.
- A lightweight random-forest ablation without intake_year was rerun on the existing processed dataset.
- Subgroup model performance by animal type reused the saved main random-forest model and the deterministic stratified split.
