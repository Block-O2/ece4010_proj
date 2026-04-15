# Extended Experiments

## Experiment Table

| Setting | Model | Accuracy | F1-score | ROC-AUC |
|---|---|---:|---:|---:|
| `fast_adoption_30d` + stratified split | `logreg` | 0.6463 | 0.5828 | 0.7313 |
| `fast_adoption_30d` + stratified split | `rf` | 0.7289 | 0.6406 | 0.8149 |
| `adoption` + stratified split | `logreg` | 0.6648 | 0.6892 | 0.7270 |
| `adoption` + stratified split | `rf` | 0.7156 | 0.7268 | 0.7961 |
| `fast_adoption_30d` + time split | `logreg` | 0.6165 | 0.5498 | 0.6724 |
| `fast_adoption_30d` + time split | `rf` | 0.6983 | 0.5773 | 0.7654 |

## Takeaways

- Random Forest is the strongest model in every setting that was tested.
- The fallback `adoption` label is easier to predict than `fast_adoption_30d`.
- Time-based evaluation is stricter than the default stratified split and lowers performance for both models.
- The project now includes one main baseline and two meaningful comparison experiments, which is enough for a stronger course presentation than a single-run baseline.
