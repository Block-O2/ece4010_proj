# Predicting 30-Day Pet Adoption from Shelter Intake Data

Machine learning project predicting whether a dog or cat from Austin Animal Center will be adopted within 30 days of intake, using public intake/outcome records and intake-time features only.

Chinese documentation: [README_zh.md](README_zh.md)

## Project Summary

This repository builds a reproducible baseline ML pipeline for shelter-adoption prediction. It downloads or loads Austin Animal Center intake/outcome CSV files, pairs intake and outcome records, constructs a `fast_adoption_30d` label, trains baseline classifiers, and reports model metrics, feature importance, and comparison experiments. The project is presented as a solid baseline analysis, not a production shelter decision system.

## Dataset

Primary data sources:

1. Austin Animal Center Intakes, official open data.
2. Austin Animal Center Outcomes, official open data.

The default download URLs are configured in [src/data_loading.py](src/data_loading.py). If automatic download fails, manually place the CSV files under `data/raw/`:

- `data/raw/austin_animal_center_intakes.csv`
- `data/raw/austin_animal_center_outcomes.csv`

Accepted fallback filenames:

- `intakes.csv`
- `outcomes.csv`
- `Austin_Animal_Center_Intakes.csv`
- `Austin_Animal_Center_Outcomes.csv`
- `Austin_Animal_Center_Intakes__10_01_2013_to_05_05_2025_.csv`
- `Austin_Animal_Center_Outcomes__10_01_2013_to_05_05_2025_.csv`

No Kaggle token is required.

## Methods

Task definition:

- Keep only `Dog` and `Cat` records.
- Match intake records with outcome records using a chronological pairing rule for repeated `Animal ID`s.
- Define `fast_adoption_30d = 1` only when:
  - `Outcome Type == "Adoption"`
  - `outcome_datetime - intake_datetime <= 30 days`
- Otherwise set `fast_adoption_30d = 0`.

Fallback label:

- `adoption = 1` if `Outcome Type == "Adoption"`.
- `adoption = 0` otherwise.

Default model target: `fast_adoption_30d`.

Only intake-time features are used as model inputs:

- `Animal Type`
- `Breed`
- `Color`
- `Sex upon Intake`
- `Age upon Intake`
- `Intake Type`
- `Intake Condition`
- intake month, weekday, season, and year

Outcome-side fields are used only for label construction and post-hoc analysis, not as model inputs.

Models:

- `logreg`: Logistic Regression baseline.
- `rf`: Random Forest main model.
- `xgb`: optional XGBoost model if installed.

## Results

Latest verified full-data run with the default target `fast_adoption_30d`:

- Dataset rows after pairing and feature construction: `163932`
- Positive rate for `fast_adoption_30d`: `0.3294`
- Positive rate for fallback `adoption`: `0.5036`
- Outcome match rate after chronological pairing: `0.99`
- Temporal mismatch rate after pairing: `0.0043`

Test-set metrics:

| Model | Accuracy | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: |
| Logistic Regression | `0.6463` | `0.5828` | `0.7313` |
| Random Forest | `0.7289` | `0.6406` | `0.8149` |

Most important Random Forest features in the latest run:

- `age_upon_intake_days`
- `intake_year`
- `color`
- `sex_upon_intake`
- `intake_type`

Additional comparison experiments:

| Experiment | Model | Accuracy | F1-score | ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| `adoption`, stratified split | `logreg` | `0.6648` | `0.6892` | `0.7270` |
| `adoption`, stratified split | `rf` | `0.7156` | `0.7268` | `0.7961` |
| `fast_adoption_30d`, time split | `logreg` | `0.6165` | `0.5498` | `0.6724` |
| `fast_adoption_30d`, time split | `rf` | `0.6983` | `0.5773` | `0.7654` |

Interpretation:

- The fallback `adoption` label is easier than `fast_adoption_30d`, as expected.
- Random Forest is the strongest tested baseline.
- Time-based splitting reduces performance compared with the default stratified split, making it a useful robustness check.

## Limitations

- This is a baseline ML project, not a production shelter decision system.
- The chronological pairing rule is simple and reproducible, but it is not a full event-reconstruction algorithm for every repeated `Animal ID` case.
- Official CSV download can fail when the public endpoint blocks scripted requests; manual CSV placement is supported by design.
- High-cardinality categories such as breed and color are grouped to keep the baseline lightweight.
- Full-data sklearn training can be memory-heavy on some laptops.
- The model should not be used for real adoption decisions without deeper fairness, leakage, policy, and deployment analysis.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows example:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.check_data --no-download
python -m src.evaluate --no-download --max-rows 50000
```

## How To Run

Build processed datasets:

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

Use local CSV files without automatic download:

```bash
python src/evaluate.py --no-download
```

Switch to the fallback label:

```bash
python src/evaluate.py --target adoption
```

Low-resource run on a sampled subset:

```bash
python src/evaluate.py --no-download --max-rows 50000
```

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

## Latest Verified Run

This repository was verified on a Windows machine on 2026-04-15.

Completed in that run:

- Cloned the repository locally on Windows.
- Created a local `.venv` and installed dependencies.
- Downloaded the official Austin Animal Center intake and outcome CSV files.
- Built `data/processed/paired_dataset.csv` and `data/processed/modeling_dataset.csv`.
- Generated `data/processed/dataset_report.json` and `data/processed/data_audit.md`.
- Ran the full evaluation suite on the complete processed dataset.
- Saved updated metrics, confusion matrices, feature-importance outputs, and model artifacts.

Compatibility notes:

- `RareCategoryGrouper` implements `get_feature_names_out`, which fixes feature-importance export with newer scikit-learn releases.
- The unused `n_jobs` argument was removed from logistic regression to avoid a scikit-learn warning on newer versions.

## Outputs

Processed data:

- `data/processed/paired_dataset.csv`
- `data/processed/modeling_dataset.csv`
- `data/processed/dataset_report.json`
- `data/processed/data_audit.md`

Training and evaluation artifacts:

- `outputs/models/*.joblib`
- `outputs/metrics/*_metrics.json`
- `outputs/metrics/model_comparison.json`
- `outputs/metrics/rf_feature_importance.csv`
- `outputs/figures/*_confusion_matrix.png`
- `outputs/figures/rf_feature_importance.png`
- [outputs/results_summary.md](outputs/results_summary.md)
- [outputs/extended_experiments.md](outputs/extended_experiments.md)

Final project materials:

- [materials_status.md](materials_status.md)
- [final_report.md](final_report.md)
- [project_summary.md](project_summary.md)
- [report_takeaways.md](report_takeaways.md)
- [future_work.md](future_work.md)
- [outputs/final_materials](outputs/final_materials)
- [outputs/final_materials/main_results_table.md](outputs/final_materials/main_results_table.md)
- [outputs/final_materials/comparison_results_table.md](outputs/final_materials/comparison_results_table.md)
- [outputs/final_materials/figures_manifest.md](outputs/final_materials/figures_manifest.md)
- [outputs/final_materials/additional_analysis.md](outputs/final_materials/additional_analysis.md)

## Matching Rule

The repository uses a simple reproducible rule for repeated `Animal ID`s:

- Sort intakes by `Animal ID` and intake datetime.
- Sort outcomes by `Animal ID` and outcome datetime.
- Pair the `k`-th intake with the `k`-th outcome for the same animal.

This avoids an unstable many-to-many merge on repeated IDs and keeps the logic easy to explain. Full notes are in [matching_strategy.md](matching_strategy.md).

## Work Completed

- Built the reusable project structure, Python modules, notebook scaffold, and proposal/slides outlines.
- Implemented raw-data loading with download fallback and manual CSV support.
- Implemented column normalization, mixed-format datetime parsing, intake/outcome pairing, and both labels.
- Verified that the main `fast_adoption_30d` task can be constructed on Austin Animal Center data.
- Added a data audit script and generated processed datasets locally.
- Added lower-resource execution options by reusing `data/processed/modeling_dataset.csv` and supporting `--max-rows`.
- Verified the full Windows workflow from environment setup to full-data evaluation.
- Generated baseline outputs and updated the repository summary with real metrics.
- Fixed scikit-learn compatibility for feature-importance export on newer environments.

## Next Steps

- Turn verified results and plots into final slides or report sections.
- Present `adoption` as a comparison experiment rather than the main claim.
- Expand the EDA notebook with presentation-ready descriptive charts if needed.
- Add XGBoost or hyperparameter tuning only after preserving the current baseline.

## Fallback Note

The primary implementation keeps the `fast_adoption_30d` target. If data inspection later shows the 30-day label is unstable because repeated-record pairing is unreliable, use the existing fallback label `adoption` while keeping the fast-adoption design clearly documented.
