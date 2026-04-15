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
- `Austin_Animal_Center_Intakes__10_01_2013_to_05_05_2025_.csv`
- `Austin_Animal_Center_Outcomes__10_01_2013_to_05_05_2025_.csv`

No Kaggle token is required.

## Project Structure

```text
ece4010_proj/
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|   `-- eda.ipynb
|-- outputs/
|   |-- figures/
|   |-- metrics/
|   |-- models/
|   `-- results_summary.md
|-- src/
|   |-- data_loading.py
|   |-- dataset_builder.py
|   |-- evaluate.py
|   |-- feature_engineering.py
|   |-- modeling.py
|   |-- preprocessing.py
|   |-- train.py
|   `-- utils.py
|-- matching_strategy.md
|-- proposal_outline.md
|-- slides_outline.md
|-- README.md
`-- requirements.txt
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

Low-resource run on a smaller sampled subset:

```bash
python src/evaluate.py --no-download --max-rows 50000
```

Continue on a Windows gaming laptop:

The current baseline is built on pandas and scikit-learn, so it is still mostly CPU / memory bound rather than GPU bound. An RTX 4060 is not the main accelerator for this version, but a better cooling system and a larger RAM budget should make the runs more stable.

Recommended Windows commands:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.check_data --no-download
python -m src.evaluate --no-download --max-rows 50000
```

## Latest Verified Run

This repository was verified on a Windows machine on 2026-04-15.

What was completed in that run:

- Cloned the repository locally on Windows
- Created a local `.venv` and installed the required dependencies
- Downloaded the official Austin Animal Center intake and outcome CSV files
- Built `data/processed/paired_dataset.csv` and `data/processed/modeling_dataset.csv`
- Generated `data/processed/dataset_report.json` and `data/processed/data_audit.md`
- Ran the full evaluation suite on the complete processed dataset
- Saved updated metrics, confusion matrices, feature-importance outputs, and model artifacts

Compatibility note:

- `RareCategoryGrouper` now implements `get_feature_names_out`, which fixes feature-importance export with newer scikit-learn releases.
- The unused `n_jobs` argument was removed from logistic regression to avoid a scikit-learn warning on newer versions.

## Latest Baseline Results

Full-data run with the default target `fast_adoption_30d`:

- Dataset rows after pairing and feature construction: `163932`
- Positive rate for `fast_adoption_30d`: `0.3294`
- Positive rate for fallback `adoption`: `0.5036`
- Outcome match rate after chronological pairing: `0.99`
- Temporal mismatch rate after pairing: `0.0043`

Test-set metrics from the latest full evaluation:

- `logreg`: Accuracy `0.6463`, F1-score `0.5828`, ROC-AUC `0.7313`
- `rf`: Accuracy `0.7289`, F1-score `0.6406`, ROC-AUC `0.8149`

Most important random-forest features in the latest run:

- `age_upon_intake_days`
- `intake_year`
- `color`
- `sex_upon_intake`
- `intake_type`

## Outputs

Processed data:

- [data/processed/paired_dataset.csv](/Users/hankli/Desktop/coding/ece4010_proj/data/processed/paired_dataset.csv)
- [data/processed/modeling_dataset.csv](/Users/hankli/Desktop/coding/ece4010_proj/data/processed/modeling_dataset.csv)
- [data/processed/dataset_report.json](/Users/hankli/Desktop/coding/ece4010_proj/data/processed/dataset_report.json)
- `data/processed/data_audit.md`

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
- Full-data sklearn training can still be heavy on laptops with limited memory on some machines, even though the verified Windows run completed successfully.

## Work Completed

- Built the reusable project structure, Python modules, notebook scaffold, and proposal/slides outlines.
- Implemented raw-data loading with download fallback and manual CSV support.
- Implemented column normalization, mixed-format datetime parsing, intake/outcome pairing, and both labels.
- Verified that the main `fast_adoption_30d` task can be constructed on the Austin Animal Center data.
- Added a data audit script and generated processed datasets locally.
- Added lower-resource execution options by reusing `data/processed/modeling_dataset.csv` and supporting `--max-rows`.
- Verified the full Windows workflow from environment setup to full-data evaluation.
- Generated a complete set of baseline outputs and updated the repository summary with real metrics.
- Fixed scikit-learn compatibility for feature-importance export on newer environments.

## Next Steps

- Review the generated confusion matrices and random-forest feature-importance plot for presentation use.
- Decide whether to keep `fast_adoption_30d` as the main report target or present `adoption` as a fallback robustness check.
- Expand the EDA notebook and presentation materials using the now-verified dataset and saved outputs.
- If desired, add time-based split experiments with `python -m src.evaluate --no-download --split-method time`.

## Fallback Note

The primary implementation keeps the `fast_adoption_30d` target. If data inspection later shows the 30-day label is unstable because repeated-record pairing is unreliable, use the existing fallback label `adoption` while keeping the fast-adoption design in the proposal and README.
