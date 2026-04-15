# Slides Outline

## Slide 1: Title

- Predicting Fast Pet Adoption from Shelter Intake Data
- Course project baseline and comparison experiments
- Team / course / date
- Austin Animal Center case study

Suggested visual:

- Shelter photo or simple title graphic

## Slide 2: Problem Motivation

- Shelters benefit from understanding which animals are likely to be adopted quickly
- We focus on a concrete binary target: adoption within 30 days of intake
- The goal is a reproducible, explainable ML baseline rather than a production deployment

Suggested visual:

- One intake-to-outcome flow diagram

## Slide 3: Prediction Task

- Input: information available at intake time only
- Main output: `fast_adoption_30d`
- Fallback output: `adoption`
- Dogs and cats only
- No outcome-only features in the model inputs

Suggested visual:

- Input / label / model pipeline diagram

## Slide 4: Dataset

- Austin Animal Center Intakes and Outcomes
- Public City of Austin open data
- Filtered rows:
  - Intakes: `163932`
  - Outcomes: `163904`
- Pairing statistics:
  - Outcome match rate: `0.99`
  - Temporal mismatch rate: `0.0043`

Suggested visual:

- Small dataset summary table

## Slide 5: Label Construction

- Match intake and outcome records by `Animal ID`
- Repeated IDs handled by chronological pairing
- `fast_adoption_30d = 1` only if outcome is adoption within 30 days
- `adoption` kept as a fallback / comparison target

Suggested visual:

- Timeline example showing the `k`-th intake paired with the `k`-th outcome

## Slide 6: Features

- Animal Type
- Breed
- Color
- Sex upon Intake
- Age upon Intake
- Intake Type
- Intake Condition
- Intake month / weekday / season / year

Suggested visual:

- Grouped feature table or intake-only feature graphic

## Slide 7: Methods

- Logistic Regression baseline
- Random Forest main model
- Main evaluation: stratified train / validation / test split
- Extra robustness check: time-based split

Suggested visual:

- Simple model-and-split comparison chart

## Slide 8: Main Results

- `fast_adoption_30d` + stratified split
- Logistic Regression:
  - Accuracy `0.6463`
  - F1-score `0.5828`
  - ROC-AUC `0.7313`
- Random Forest:
  - Accuracy `0.7289`
  - F1-score `0.6406`
  - ROC-AUC `0.8149`

Suggested visual:

- Metric comparison table

## Slide 9: Comparison Experiments

- Fallback target `adoption`
  - `logreg`: Accuracy `0.6648`, F1 `0.6892`, ROC-AUC `0.7270`
  - `rf`: Accuracy `0.7156`, F1 `0.7268`, ROC-AUC `0.7961`
- Time-based split on `fast_adoption_30d`
  - `logreg`: Accuracy `0.6165`, F1 `0.5498`, ROC-AUC `0.6724`
  - `rf`: Accuracy `0.6983`, F1 `0.5773`, ROC-AUC `0.7654`

Suggested visual:

- One compact experiment table with three settings

## Slide 10: Interpretation

- Random Forest is the strongest model in every setting
- `adoption` is easier to predict than `fast_adoption_30d`
- Time-based evaluation is stricter and lowers performance
- Age and intake context matter strongly

Suggested visual:

- Random-forest feature importance plot

## Slide 11: Limitations

- Repeated `Animal ID`s use a chronological heuristic rather than full event reconstruction
- Rare-category grouping simplifies breed and color information
- This is a course-project baseline, not a production shelter decision tool

Suggested visual:

- Risk / limitation table

## Slide 12: Final Takeaway

- The project now includes a verified data pipeline, completed baseline experiments, and report-ready outputs
- The main baseline is reproducible and performs meaningfully better with Random Forest than with Logistic Regression
- Next extensions could include XGBoost, richer EDA, or more formal report polishing

Suggested visual:

- Final summary slide with 3 takeaways
