# Future Work

- Add XGBoost as a lightweight third tabular baseline and compare it against the current Random Forest without redesigning the project.
- Replace the chronological pairing heuristic with a more careful event-reconstruction strategy for repeated `Animal ID` records if time allows.
- Expand the EDA notebook with a small number of polished report-ready figures on age, intake type, and animal-type differences.
- Explore calibration analysis so the predicted probabilities are more interpretable as risk scores rather than only ranking signals.
- Evaluate whether a smaller, simpler feature set can preserve most of the performance while improving interpretability.
- Test a few targeted hyperparameter adjustments for Random Forest instead of adding a large tuning pipeline.
- Compare more fairness- and subgroup-oriented slices beyond dog versus cat, such as intake type or age band.
- If the course expects stronger temporal realism, emphasize the time-based split further or add rolling-time validation.
