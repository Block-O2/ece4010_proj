# Report Takeaways

- The project uses only intake-time information to predict whether a dog or cat will be adopted within 30 days, which keeps the task realistic and avoids outcome-side leakage.
- A simple chronological pairing rule for repeated `Animal ID` records makes the public intake and outcome tables usable for supervised learning without unstable many-to-many merges.
- On the verified full-data stratified split, Random Forest clearly outperforms Logistic Regression on the main `fast_adoption_30d` task: ROC-AUC 0.8149 versus 0.7313.
- The fallback `adoption` target is easier than the stricter 30-day target, which supports the idea that `fast_adoption_30d` is the more challenging and interesting prediction task.
- Performance drops under a time-based split, so the random stratified baseline should be interpreted as an optimistic but useful reference rather than a future-proof estimate.
- Age at intake is the single strongest feature in the main random-forest model, and intake context variables such as year, sex, color, and intake type also matter.
- Removing `intake_year` causes only a modest decline in random-forest performance, suggesting that the model is not driven by year alone.
- The project is strongest as a reproducible, explainable tabular-ML baseline for a course setting, not as a production shelter decision system.
