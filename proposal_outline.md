# Proposal / Report Outline

## Problem Statement

Predict whether a dog or cat entering the Austin Animal Center will be adopted within 30 days, using only information available at intake time.

## Motivation

- Fast adoption is a concrete, easy-to-explain shelter outcome.
- Early prediction can help identify animals that may need more promotion or intervention.
- The task is a clean binary classification problem that fits a course-level machine-learning project.

## Dataset

- Austin Animal Center Intakes
- Austin Animal Center Outcomes
- Public City of Austin open-data records
- Focus on dogs and cats only
- Label built by pairing intake and outcome records and checking whether the outcome is `Adoption` within 30 days

Verified dataset summary from the latest run:

- Intake rows after filtering: `163932`
- Outcome rows after filtering: `163904`
- Outcome match rate after pairing: `0.99`
- Temporal mismatch rate: `0.0043`
- Main-target positive rate (`fast_adoption_30d`): `0.3294`
- Fallback-target positive rate (`adoption`): `0.5036`

## Method

- Standardize raw column names and parse mixed-format datetimes
- Keep only intake-time features to avoid data leakage
- Convert age strings into numeric day counts
- Engineer calendar features from intake datetime
- Collapse rare breed and color categories into `Other`
- Match repeated `Animal ID`s using chronological pairing
- Train and compare:
  - Logistic Regression baseline
  - Random Forest main model
  - Optional XGBoost for future extension

## Main Results

Main evaluation setting: `fast_adoption_30d` with stratified split

- Logistic Regression: Accuracy `0.6463`, F1-score `0.5828`, ROC-AUC `0.7313`
- Random Forest: Accuracy `0.7289`, F1-score `0.6406`, ROC-AUC `0.8149`

Main interpretation:

- Random Forest clearly outperforms Logistic Regression
- Non-linear interactions among age, intake type, sex, color, and calendar effects appear important
- The project already provides a strong and reproducible baseline rather than just a proposal

## Comparison Experiments

Fallback task: `adoption` with stratified split

- Logistic Regression: Accuracy `0.6648`, F1-score `0.6892`, ROC-AUC `0.7270`
- Random Forest: Accuracy `0.7156`, F1-score `0.7268`, ROC-AUC `0.7961`

Time-based robustness check: `fast_adoption_30d` with time split

- Logistic Regression: Accuracy `0.6165`, F1-score `0.5498`, ROC-AUC `0.6724`
- Random Forest: Accuracy `0.6983`, F1-score `0.5773`, ROC-AUC `0.7654`

What these comparisons show:

- The fallback `adoption` label is easier to predict than the stricter 30-day task
- Random Forest is consistently the strongest model
- Time-based splitting lowers performance, which makes it a more realistic but stricter evaluation setting

## Feature Insights

Top random-forest features in the verified baseline run:

1. `age_upon_intake_days`
2. `intake_year`
3. `color`
4. `sex_upon_intake`
5. `intake_type`

These results suggest that age and intake context matter strongly for short-term adoption outcomes.

## Feasibility

- Data is public and already verified locally
- The pipeline uses standard Python ML tools: pandas and scikit-learn
- The full workflow now runs successfully on Windows with a local virtual environment
- Outputs already include metrics, plots, saved models, and summary markdown files

## Risks And Limitations

- Repeated `Animal ID`s are handled with a simple chronological heuristic, not full event reconstruction
- Rare category grouping simplifies the modeling task but removes some detail
- The repository is a course-project baseline, not a production deployment

## Final Positioning

This project is no longer just a proposal skeleton. It now contains:

- a verified data pipeline
- completed baseline experiments
- two comparison experiments
- reproducible outputs for slides or a written report
