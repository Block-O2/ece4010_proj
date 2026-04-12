from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .dataset_builder import build_dataset
    from .modeling import create_data_splits, train_and_evaluate_model
    from .utils import ensure_directory, set_global_seed
except ImportError:
    from dataset_builder import build_dataset
    from modeling import create_data_splits, train_and_evaluate_model
    from utils import ensure_directory, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one baseline model for fast pet adoption.")
    parser.add_argument("--model", choices=["logreg", "rf", "xgb"], default="rf")
    parser.add_argument("--target", choices=["fast_adoption_30d", "adoption"], default="fast_adoption_30d")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--split-method", choices=["stratified", "time"], default="stratified")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not attempt to download data automatically.",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    ensure_directory(Path(args.outputs_dir))

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
    artifact = train_and_evaluate_model(
        df_splits=splits,
        model_name=args.model,
        target_column=args.target,
        outputs_dir=args.outputs_dir,
        seed=args.seed,
    )

    print(f"Finished training {args.model} on target `{args.target}`.")
    print(f"Test metrics: {artifact['test_metrics']}")


if __name__ == "__main__":
    main()
