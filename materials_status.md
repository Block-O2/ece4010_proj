# Materials Status

This file audits the repository as a final-project materials package rather than as a baseline-only repository.

## Already Present And Usable

- Core code pipeline in `src/`
  - raw data loading
  - intake/outcome standardization
  - chronological pairing
  - target construction for `fast_adoption_30d` and `adoption`
  - feature engineering
  - logistic-regression and random-forest training / evaluation
- Verified processed datasets
  - `data/processed/paired_dataset.csv`
  - `data/processed/modeling_dataset.csv`
  - `data/processed/dataset_report.json`
  - `data/processed/data_audit.md`
- Verified main experiment outputs
  - `outputs/metrics/logreg_metrics.json`
  - `outputs/metrics/rf_metrics.json`
  - `outputs/metrics/model_comparison.json`
  - `outputs/metrics/rf_feature_importance.csv`
  - `outputs/figures/logreg_confusion_matrix.png`
  - `outputs/figures/rf_confusion_matrix.png`
  - `outputs/figures/rf_feature_importance.png`
  - `outputs/models/logreg_model.joblib`
  - `outputs/models/rf_model.joblib`
- Comparison experiment outputs already present locally
  - `outputs_adoption/` for the fallback-label experiment
  - `outputs_time/` for the time-based-split experiment
- Existing documentation that can be reused directly
  - `README.md`
  - `README_zh.md`
  - `matching_strategy.md`
  - `proposal_outline.md`
  - `slides_outline.md`
  - `outputs/results_summary.md`
  - `outputs/extended_experiments.md`
- Existing lightweight notebook
  - `notebooks/eda.ipynb`

## Missing Before This Final-Materials Pass

These items were not present as polished final-project materials before this pass:

- `final_report.md`
- `project_summary.md`
- `report_takeaways.md`
- `future_work.md`
- `outputs/final_materials/`
- curated final tables for report use
- curated final figure set with report-ready filenames
- `figures_manifest.md`
- a compact final additional-analysis writeup
- a clear audit explaining what did and did not need rerunning

## Reusable Directly

- Main metrics from `outputs/metrics/model_comparison.json`
- Fallback-label metrics from `outputs_adoption/metrics/model_comparison.json`
- Time-split metrics from `outputs_time/metrics/model_comparison.json`
- Main confusion matrices from `outputs/figures/`
- Main random-forest feature importance from `outputs/metrics/rf_feature_importance.csv`
- Processed modeling dataset from `data/processed/modeling_dataset.csv`
- Dataset statistics from `data/processed/dataset_report.json`

## Needed Regeneration Or New Creation

- Final report-style markdown documents
- Final tables in markdown and CSV form
- Curated final-materials figures with clean filenames
- Figure manifest for later report / slide use
- One lightweight additional analysis package
  - subgroup analysis by animal type
  - random-forest ablation without `intake_year`

## What Did Not Need To Be Rerun

- The full main baseline evaluation
- Raw-data download and pairing pipeline
- Main logistic-regression training
- Main random-forest training
- Existing fallback-label experiment
- Existing time-split experiment

## What Was Rerun In A Targeted Way

- A lightweight random-forest ablation that removes `intake_year`
  - rerun only because it did not already exist
- Final-materials plotting and table generation from already processed data and already saved metrics

## Honest Notes

- The repository already had the core scientific content for a strong baseline project.
- The main gap was not missing code correctness, but missing final report materials and curated presentation-ready outputs.
- The `outputs_adoption/` and `outputs_time/` experiment directories exist locally and are useful for reuse, but they are not intended to be committed as large derived-output directories.
