from __future__ import annotations

import argparse
from pathlib import Path

from dataset_builder import build_dataset
from modeling import create_data_splits, train_and_evaluate_model
from utils import ensure_directory, save_json, set_global_seed


def build_results_summary(
    comparison: dict[str, dict],
    output_path: str | Path,
) -> None:
    rf_metrics = comparison.get("rf", {})
    logreg_metrics = comparison.get("logreg", {})
    rf_feature_importance = rf_metrics.get("feature_importance", [])
    top_features = ", ".join(item["feature"] for item in rf_feature_importance[:5]) or "feature importance unavailable"

    logreg_test = logreg_metrics.get("test_metrics", {})
    rf_test = rf_metrics.get("test_metrics", {})

    summary_text = f"""This baseline predicts whether a dog or cat will be adopted within 30 days of intake using only intake-time information. The evaluation compares a logistic regression baseline with a random forest main model on the same held-out split, so the results are directly comparable. Logistic regression reached Accuracy {logreg_test.get('accuracy')}, F1-score {logreg_test.get('f1_score')}, and ROC-AUC {logreg_test.get('roc_auc')}. Random forest reached Accuracy {rf_test.get('accuracy')}, F1-score {rf_test.get('f1_score')}, and ROC-AUC {rf_test.get('roc_auc')}.

The gap between the two models indicates how much non-linear structure exists in the shelter intake features. If random forest performs better, that suggests interactions among breed, intake condition, age, and calendar effects matter beyond a linear baseline. If the two models are close, that is still a useful course-project result because it shows a simple interpretable model is already competitive for this task. The most influential random forest features in the current run were {top_features}.

These results should be presented as a reproducible baseline rather than a final production system. The main limitations are the simple chronological pairing rule for repeated Animal IDs, the grouping of rare breed and color categories into `Other`, and the fact that outcome-side information is intentionally excluded from the input to avoid leakage. Those constraints are appropriate for a clear, feasible machine learning course project."""

    Path(output_path).write_text(summary_text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full baseline evaluation suite.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", choices=["fast_adoption_30d", "adoption"], default="fast_adoption_30d")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--split-method", choices=["stratified", "time"], default="stratified")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "rf"],
        choices=["logreg", "rf", "xgb"],
        help="Models to run in the evaluation suite.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not attempt to download data automatically.",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    outputs_dir = ensure_directory(args.outputs_dir)

    dataset_result = build_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        download=not args.no_download,
    )
    modeling_df = dataset_result["modeling_df"]
    splits = create_data_splits(
        modeling_df,
        target_column=args.target,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        split_method=args.split_method,
    )

    comparison: dict[str, dict] = {}
    for model_name in args.models:
        comparison[model_name] = train_and_evaluate_model(
            df_splits=splits,
            model_name=model_name,
            target_column=args.target,
            outputs_dir=outputs_dir,
            seed=args.seed,
        )

    save_json(comparison, Path(outputs_dir) / "metrics" / "model_comparison.json")
    build_results_summary(comparison, Path(outputs_dir) / "results_summary.md")

    print("Completed evaluation suite.")
    for model_name, artifact in comparison.items():
        print(f"{model_name}: {artifact['test_metrics']}")


if __name__ == "__main__":
    main()
