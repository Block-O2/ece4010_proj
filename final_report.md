# Predicting Fast Pet Adoption from Shelter Intake Data

## 1. Title

Predicting whether a shelter dog or cat will be adopted within 30 days using Austin Animal Center intake data.

## 2. Abstract

This project studies whether public shelter intake records can be used to predict fast adoption outcomes for dogs and cats. Using Austin Animal Center open data, we construct a binary target `fast_adoption_30d` that marks whether an animal is adopted within 30 days of intake. Because the public data contains repeated `Animal ID` values, we use a simple chronological pairing rule to match intake and outcome records in a reproducible way. We train two lightweight tabular baselines, Logistic Regression and Random Forest, using only information available at intake time to avoid data leakage. On the verified full-data stratified split, Logistic Regression reaches Accuracy 0.6463, F1-score 0.5828, and ROC-AUC 0.7313, while Random Forest reaches Accuracy 0.7289, F1-score 0.6406, and ROC-AUC 0.8149. We also include two comparison experiments: a fallback `adoption` target and a time-based split. Finally, we add a small ablation without `intake_year` and a subgroup analysis by animal type to strengthen the final project package.

## 3. Introduction / Motivation

Animal shelters often face limited resources and large numbers of animals entering the system over time. A simple predictive model that estimates whether an animal is likely to be adopted quickly could help staff think about promotion, outreach, and resource allocation. For a course project, this setting is especially attractive because it uses public data, defines a clear binary classification task, and produces results that are easy to interpret.

The goal of this repository is not to create a production shelter decision system. Instead, it aims to deliver a clean, reproducible machine-learning baseline that can be defended in a course presentation or final report. That makes transparency and reproducibility just as important as raw predictive performance.

## 4. Problem Definition

The main task is:

- Input: information available at intake time only
- Output: whether a dog or cat is adopted within 30 days of intake

Formally, the main label is:

- `fast_adoption_30d = 1` if `Outcome Type == "Adoption"` and `outcome_datetime - intake_datetime <= 30 days`
- `fast_adoption_30d = 0` otherwise

The repository also keeps a fallback label:

- `adoption = 1` if `Outcome Type == "Adoption"`
- `adoption = 0` otherwise

This fallback target is useful because it tests whether the 30-day target is substantially harder than overall adoption prediction.

## 5. Dataset and Label Construction

The project uses two public datasets from the Austin Animal Center:

1. Austin Animal Center Intakes
2. Austin Animal Center Outcomes

The pipeline standardizes column names, parses mixed-format datetimes, filters to dogs and cats, and then constructs paired intake/outcome records.

Verified dataset summary from the latest full run:

- Filtered intake rows: 163932
- Filtered outcome rows: 163904
- Unique intake `Animal ID`s: 146450
- Unique outcome `Animal ID`s: 146409
- Rows with matched outcome after pairing: 162285
- Outcome match rate: 0.99
- Temporal mismatch rate: 0.0043
- Positive rate for `fast_adoption_30d`: 0.3294
- Positive rate for `adoption`: 0.5036

The processed datasets are saved as:

- `data/processed/paired_dataset.csv`
- `data/processed/modeling_dataset.csv`

## 6. Matching Strategy

The most important data-engineering design choice in this project is how intake and outcome records are paired when the same `Animal ID` appears multiple times. A direct merge on `Animal ID` would create unstable many-to-many joins and unreliable labels.

To keep the project simple, reproducible, and explainable, we use chronological pairing:

1. Sort intake records by `Animal ID` and `intake_datetime`
2. Sort outcome records by `Animal ID` and `outcome_datetime`
3. Assign `pairing_sequence = 0, 1, 2, ...` within each animal
4. Match the `k`-th intake with the `k`-th outcome for that animal

This strategy does not attempt full event reconstruction. Instead, it provides a stable baseline rule that is easy to implement and easy to explain in a course setting.

If a matched outcome appears earlier than the intake timestamp, the pipeline marks the pair as a temporal mismatch and removes the outcome side from label construction for that row. This prevents obviously invalid timestamps from contaminating the target.

## 7. Features and Preprocessing

Only intake-time features are used as model inputs:

- `animal_type`
- `breed`
- `color`
- `sex_upon_intake`
- `age_upon_intake_days`
- `intake_type`
- `intake_condition`
- `intake_month`
- `intake_weekday`
- `intake_season`
- `intake_year`

Preprocessing steps include:

- column-name normalization
- text-category cleanup
- conversion of age strings such as `"2 years"` into numeric days
- calendar feature extraction from intake datetime
- rare-category grouping for high-cardinality fields such as breed and color

The project deliberately excludes outcome-side fields from model inputs to avoid leakage.

## 8. Models

The repository focuses on lightweight tabular baselines:

- Logistic Regression as a simple linear baseline
- Random Forest as the main non-linear model

An optional XGBoost path exists in the codebase for future work, but it is not required for the current project package.

This model choice fits the project goals well:

- standard tools
- fast to run
- easy to interpret
- appropriate for a course project

## 9. Experimental Setup

The main evaluation uses a stratified train / validation / test split with seed 42. This keeps the label balance stable across splits.

Main evaluation target:

- `fast_adoption_30d`

Comparison settings:

1. fallback target `adoption`
2. time-based split for `fast_adoption_30d`

The full-data baseline was already verified locally and reused here. To strengthen the final project materials without restarting the project, only lightweight extra analyses were added:

- random-forest ablation without `intake_year`
- subgroup analysis by animal type

## 10. Main Results

Main evaluation setting: `fast_adoption_30d` with stratified split

| Model | Accuracy | F1-score | ROC-AUC |
| --- | --- | --- | --- |
| Logistic Regression | 0.6463 | 0.5828 | 0.7313 |
| Random Forest | 0.7289 | 0.6406 | 0.8149 |

Random Forest clearly outperforms Logistic Regression across all three main metrics. This suggests that the relationship between intake features and fast adoption is not purely linear. In particular, interactions among age, intake type, calendar effects, and categorical shelter metadata appear to matter.

The strongest random-forest features in the verified main run were:

1. `age_upon_intake_days`
2. `intake_year`
3. `color`
4. `sex_upon_intake`
5. `intake_type`

## 11. Comparison Experiments

### 11.1 Fallback Target: `adoption`

| Model | Accuracy | F1-score | ROC-AUC |
| --- | --- | --- | --- |
| Logistic Regression | 0.6648 | 0.6892 | 0.7270 |
| Random Forest | 0.7156 | 0.7268 | 0.7961 |

This comparison shows that overall adoption is easier to predict than adoption within 30 days. That is intuitive, because `adoption` is a broader and less restrictive target.

### 11.2 Time-Based Split: `fast_adoption_30d`

| Model | Accuracy | F1-score | ROC-AUC |
| --- | --- | --- | --- |
| Logistic Regression | 0.6165 | 0.5498 | 0.6724 |
| Random Forest | 0.6983 | 0.5773 | 0.7654 |

This experiment is stricter than the default stratified split because it more closely resembles predicting future data rather than a random held-out subset from the same distribution. Performance drops for both models, which is expected and makes this a useful robustness check.

### 11.3 What The Comparisons Mean

- Random Forest is the strongest model in every tested setting
- `adoption` is easier than `fast_adoption_30d`
- time-based evaluation is harsher and more realistic than a stratified random split

## 12. Additional Analysis

Two lightweight additions were made to make the project feel more complete without turning it into a new modeling branch.

### 12.1 Subgroup Analysis By Animal Type

Descriptive differences between cats and dogs:

| animal_type | rows | fast_adoption_rate | adoption_rate | median_age_days |
| --- | --- | --- | --- | --- |
| Cat | 69324 | 0.2783 | 0.5109 | 60.0 |
| Dog | 94608 | 0.3668 | 0.4982 | 365.0 |

This suggests that dogs have a higher 30-day adoption rate in this processed dataset, while cats are younger on average at intake.

The saved main random-forest model was also evaluated separately on the deterministic stratified test split:

| animal_type | accuracy | f1_score | support | positive_rate | roc_auc |
| --- | --- | --- | --- | --- | --- |
| Cat | 0.7829 | 0.6497 | 13906 | 0.2774 | 0.8657 |
| Dog | 0.6892 | 0.6358 | 18881 | 0.3677 | 0.7712 |

This indicates that subgroup behavior is not identical across animal types, and that the model appears to separate cat outcomes more cleanly by ROC-AUC even though the cat positive rate is lower.

### 12.2 Random-Forest Ablation Without `intake_year`

The most important model feature in the main run is age, but `intake_year` also ranks highly. To test whether the model depends too heavily on year, we ran a lightweight random-forest ablation that removes `intake_year` from the feature set.

Main baseline Random Forest:

- Accuracy: 0.7289
- F1-score: 0.6406
- ROC-AUC: 0.8149

Ablation Random Forest without `intake_year`:

- Accuracy: 0.7151
- F1-score: 0.6283
- ROC-AUC: 0.8014

Change vs baseline:

- Accuracy delta: -0.0138
- F1-score delta: -0.0123
- ROC-AUC delta: -0.0135

This is a noticeable but not catastrophic drop. The result suggests that `intake_year` is genuinely useful, but the model is not relying on it as the only meaningful signal.

## 13. Discussion

The project successfully demonstrates a complete course-level machine-learning pipeline from public raw data to reproducible results. Several patterns stand out:

- A non-linear tabular model performs much better than a simple linear baseline
- The stricter 30-day target is harder than overall adoption
- Time-based evaluation reduces performance, which reinforces the need for caution when interpreting random-split results
- Age and intake context are central predictive features

At the same time, this remains a baseline repository. The project is strongest when presented as a careful and explainable first system rather than as a final answer to shelter decision-making.

## 14. Limitations

- Repeated `Animal ID` records are handled with a chronological heuristic, not a full event-reconstruction algorithm
- Breed and color are simplified through rare-category grouping
- The project excludes potentially useful post-intake information by design, because the goal is intake-time prediction only
- The reported results depend on publicly available shelter records, whose collection process may itself contain noise or policy changes over time
- Time-based evaluation shows that future generalization is harder than the default random-split baseline

## 15. Ethical / Practical Considerations

This project should not be interpreted as a production deployment recommendation. Shelter outcomes are influenced by policy, staff behavior, foster availability, community outreach, and social context. A model like this can be useful for descriptive understanding or lightweight prioritization experiments, but it should not be used as an automated decision-maker about animal value or care.

There is also a fairness concern: if a model predicts that some animals are less likely to be adopted quickly, that prediction could be misused to deprioritize them. In a real shelter context, such predictions should instead be used to trigger extra support, not reduced support.

## 16. Conclusion

This repository now provides a complete final-project materials package built on top of the verified baseline. Using Austin Animal Center intake and outcome data, it predicts 30-day adoption outcomes for dogs and cats through a transparent chronological pairing pipeline and intake-only features. Random Forest is the strongest baseline in the main setting and remains strongest in both fallback-label and time-split comparisons. Additional subgroup and ablation analyses strengthen the final package without changing the project’s original scope.

Overall, the project succeeds as a lightweight, explainable, reproducible tabular machine-learning study with enough depth for a strong course submission.

## 17. Reproducibility Notes

- Main data-building entry point: `python -m src.dataset_builder`
- Main evaluation entry point: `python -m src.evaluate --no-download`
- Fallback-target comparison: `python -m src.evaluate --no-download --target adoption --outputs-dir outputs_adoption`
- Time-based comparison: `python -m src.evaluate --no-download --split-method time --outputs-dir outputs_time`
- Final materials generation: `python -m src.generate_final_materials`

Key artifacts:

- processed data in `data/processed/`
- main metrics in `outputs/metrics/`
- comparison metrics in `outputs_adoption/metrics/` and `outputs_time/metrics/`
- final tables and figures in `outputs/final_materials/`

The full main baseline did not need to be rerun for this final-materials pass. Only lightweight additional analysis was rerun on top of the existing processed dataset.
