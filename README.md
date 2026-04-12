# Predicting Fast Pet Adoption from Shelter Intake Data

Course-project baseline for predicting whether a shelter animal will be adopted within 30 days of intake using Austin Animal Center public data.

Chinese documentation: [README_zh.md](/Users/hankli/Desktop/coding/ece4010_proj/README_zh.md)

## Task Definition

Primary task:

- Keep only `Dog` and `Cat`
- Match intake records with outcome records using a simple chronological pairing rule
- Define `fast_adoption_30d = 1` only when:
  - `Outcome Type == "Adoption"`
  - `outcome_datetime - intake_datetime <= 30 days`
- Otherwise set `fast_adoption_30d = 0`

Fallback task:

- `adoption = 1` if `Outcome Type == "Adoption"`
- `adoption = 0` otherwise

The repository keeps both labels, but the default training target is `fast_adoption_30d`.

## Data Sources

Primary source:

1. Austin Animal Center Intakes (official open data)
2. Austin Animal Center Outcomes (official open data)

Default download URLs are configured in [src/data_loading.py](/Users/hankli/Desktop/coding/ece4010_proj/src/data_loading.py).

## Raw Data Placement

The code first tries to download the CSV files from the public Austin open data endpoints. If that fails, place the files manually under [data/raw](/Users/hankli/Desktop/coding/ece4010_proj/data/raw):

- `data/raw/austin_animal_center_intakes.csv`
- `data/raw/austin_animal_center_outcomes.csv`

Accepted fallback filenames are also handled:

- `intakes.csv`
- `outcomes.csv`
- `Austin_Animal_Center_Intakes.csv`
- `Austin_Animal_Center_Outcomes.csv`

No Kaggle token is required.

## Project Structure

```text
ece4010_proj/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── results_summary.md
├── src/
│   ├── data_loading.py
│   ├── dataset_builder.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── matching_strategy.md
├── proposal_outline.md
├── slides_outline.md
├── README.md
└── requirements.txt
```

## Features

Only intake-time features are used as model inputs:

- `Animal Type`
- `Breed`
- `Color`
- `Sex upon Intake`
- `Age upon Intake`
- `Intake Type`
- `Intake Condition`
- Intake datetime derived fields:
  - `month`
  - `weekday`
  - `season`
  - `year`

Outcome-side fields are used only for label construction and post-hoc analysis, not as model inputs.

## Models

- `logreg`: Logistic Regression baseline
- `rf`: Random Forest main model
- `xgb`: Optional XGBoost model if the package is installed

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How To Run

Build the processed datasets:

```bash
python src/dataset_builder.py
```

Train logistic regression:

```bash
python src/train.py --model logreg
```

Train random forest:

```bash
python src/train.py --model rf
```

Run the full evaluation suite:

```bash
python src/evaluate.py
```

If you already placed CSV files in `data/raw/` and do not want automatic download:

```bash
python src/evaluate.py --no-download
```

To switch to the fallback label:

```bash
python src/evaluate.py --target adoption
```

## Outputs

Processed data:

- [data/processed/paired_dataset.csv](/Users/hankli/Desktop/coding/ece4010_proj/data/processed/paired_dataset.csv)
- [data/processed/modeling_dataset.csv](/Users/hankli/Desktop/coding/ece4010_proj/data/processed/modeling_dataset.csv)
- [data/processed/dataset_report.json](/Users/hankli/Desktop/coding/ece4010_proj/data/processed/dataset_report.json)

Training and evaluation artifacts:

- `outputs/models/*.joblib`
- `outputs/metrics/*_metrics.json`
- `outputs/metrics/model_comparison.json`
- `outputs/metrics/rf_feature_importance.csv`
- `outputs/figures/*_confusion_matrix.png`
- `outputs/figures/rf_feature_importance.png`
- [outputs/results_summary.md](/Users/hankli/Desktop/coding/ece4010_proj/outputs/results_summary.md)

## Matching Rule

The repository uses a simple reproducible rule for repeated `Animal ID`s:

- Sort intakes by `Animal ID` and intake datetime
- Sort outcomes by `Animal ID` and outcome datetime
- Pair the `k`-th intake with the `k`-th outcome for the same animal

This avoids an unstable many-to-many merge on repeated IDs and keeps the logic easy to explain in a course proposal. Full notes are in [matching_strategy.md](/Users/hankli/Desktop/coding/ece4010_proj/matching_strategy.md).

## Current Limitations

- Official CSV download may fail because the public endpoint sometimes blocks scripted requests; manual CSV placement is supported by design.
- Repeated `Animal ID` records are handled with a simple chronological pairing heuristic, not a complex event-reconstruction algorithm.
- High-cardinality categories such as breed and color are grouped so the baseline remains lightweight and presentation-friendly.
- This repository is a baseline ML project, not a production shelter decision system.

## Fallback Note

The primary implementation keeps the `fast_adoption_30d` target. If data inspection later shows the 30-day label is unstable because repeated-record pairing is unreliable, use the existing fallback label `adoption` while keeping the fast-adoption design in the proposal and README.
