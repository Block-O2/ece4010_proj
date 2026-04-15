# Results Summary

## Main Baseline: `fast_adoption_30d` with Stratified Split

This repository predicts whether a dog or cat will be adopted within 30 days of intake using only intake-time information. The default split is stratified train/validation/test, which keeps the class balance stable across splits and provides the main baseline reported in the README.

Latest verified full-data results:

- Logistic Regression: Accuracy `0.6463`, F1-score `0.5828`, ROC-AUC `0.7313`
- Random Forest: Accuracy `0.7289`, F1-score `0.6406`, ROC-AUC `0.8149`

The random forest clearly outperforms the logistic-regression baseline on all three headline metrics. That suggests the shelter-intake features contain meaningful non-linear interactions that a simple linear decision boundary cannot capture well.

The strongest random-forest features in the verified run were:

1. `age_upon_intake_days`
2. `intake_year`
3. `color`
4. `sex_upon_intake`
5. `intake_type`

## Supporting Comparison Experiments

Two additional experiments were run to make the project more presentation-ready and to test robustness:

1. Fallback target `adoption` with the same stratified split
2. Main target `fast_adoption_30d` with a time-based split

Results:

- `adoption` + Logistic Regression: Accuracy `0.6648`, F1-score `0.6892`, ROC-AUC `0.7270`
- `adoption` + Random Forest: Accuracy `0.7156`, F1-score `0.7268`, ROC-AUC `0.7961`
- `fast_adoption_30d` + time split + Logistic Regression: Accuracy `0.6165`, F1-score `0.5498`, ROC-AUC `0.6724`
- `fast_adoption_30d` + time split + Random Forest: Accuracy `0.6983`, F1-score `0.5773`, ROC-AUC `0.7654`

These comparisons support three useful conclusions:

- The fallback `adoption` target is easier than `fast_adoption_30d`, which is reasonable because overall adoption is less strict than adoption within 30 days.
- Random Forest remains the best model in every experiment, so the model choice is stable rather than accidental.
- Performance drops under time-based splitting, which is expected because predicting future shelter outcomes is harder than predicting a random held-out subset from the same distribution.

## Project Interpretation

This repository should be presented as a reproducible course-project baseline rather than a production shelter decision system. Its main strengths are a clean public dataset, a transparent pairing rule, leakage-aware feature construction, and strong baseline comparisons.

The main limitations remain:

- repeated `Animal ID` values are handled with a chronological pairing heuristic
- rare breed and color categories are grouped into `Other`
- outcome-side information is intentionally excluded from the model inputs

Those choices are acceptable for a clear, feasible machine-learning project and are easy to explain in a proposal, slides, or course report.
