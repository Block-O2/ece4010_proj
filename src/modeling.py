from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .feature_engineering import MODEL_FEATURE_COLUMNS
    from .preprocessing import RareCategoryGrouper
    from .utils import ensure_directory, save_json
except ImportError:
    from feature_engineering import MODEL_FEATURE_COLUMNS
    from preprocessing import RareCategoryGrouper
    from utils import ensure_directory, save_json


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def create_data_splits(
    df: pd.DataFrame,
    target_column: str,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
    split_method: str = "stratified",
) -> dict[str, pd.DataFrame]:
    if split_method not in {"stratified", "time"}:
        raise ValueError("split_method must be either 'stratified' or 'time'.")

    if split_method == "time":
        ordered = df.sort_values("intake_datetime").reset_index(drop=True)
        n_rows = len(ordered)
        test_start = int(n_rows * (1 - test_size))
        val_start = int(n_rows * (1 - test_size - val_size))
        train_df = ordered.iloc[:val_start].copy()
        val_df = ordered.iloc[val_start:test_start].copy()
        test_df = ordered.iloc[test_start:].copy()
        return {"train": train_df, "val": val_df, "test": test_df}

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_column],
    )

    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_df[target_column],
    )

    return {"train": train_df.copy(), "val": val_df.copy(), "test": test_df.copy()}


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_columns = ["age_upon_intake_days"]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    categorical_pipeline = Pipeline(
        steps=[
            ("rare", RareCategoryGrouper(min_frequency=0.01, max_categories=25)),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_columns),
            ("numeric", numeric_pipeline, numeric_columns),
        ]
    )


def get_model(model_name: str, seed: int = 42) -> Any:
    if model_name == "logreg":
        return LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is optional and not installed. Install `xgboost` first, "
                "or use `--model logreg` / `--model rf`."
            ) from exc

        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
        )

    raise ValueError("model_name must be one of: logreg, rf, xgb")


def build_training_pipeline(model_name: str, seed: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(MODEL_FEATURE_COLUMNS)),
            ("classifier", get_model(model_name, seed=seed)),
        ]
    )


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> dict[str, Any]:
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    if y_score is not None and len(pd.Series(y_true).unique()) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
    else:
        metrics["roc_auc"] = None
    return metrics


def train_and_evaluate_model(
    df_splits: dict[str, pd.DataFrame],
    model_name: str,
    target_column: str,
    outputs_dir: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    outputs_path = ensure_directory(outputs_dir)
    figures_dir = ensure_directory(outputs_path / "figures")
    metrics_dir = ensure_directory(outputs_path / "metrics")
    models_dir = ensure_directory(outputs_path / "models")

    train_df = df_splits["train"]
    val_df = df_splits["val"]
    test_df = df_splits["test"]

    pipeline = build_training_pipeline(model_name, seed=seed)
    pipeline.fit(train_df[MODEL_FEATURE_COLUMNS], train_df[target_column])

    val_pred = pipeline.predict(val_df[MODEL_FEATURE_COLUMNS])
    test_pred = pipeline.predict(test_df[MODEL_FEATURE_COLUMNS])

    val_score = pipeline.predict_proba(val_df[MODEL_FEATURE_COLUMNS])[:, 1]
    test_score = pipeline.predict_proba(test_df[MODEL_FEATURE_COLUMNS])[:, 1]

    artifact = {
        "model_name": model_name,
        "target_column": target_column,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "validation_metrics": evaluate_predictions(val_df[target_column], val_pred, val_score),
        "test_metrics": evaluate_predictions(test_df[target_column], test_pred, test_score),
        "classification_report": classification_report(
            test_df[target_column],
            test_pred,
            output_dict=True,
            zero_division=0,
        ),
        "class_balance": {
            "train_positive_rate": round(float(train_df[target_column].mean()), 4),
            "val_positive_rate": round(float(val_df[target_column].mean()), 4),
            "test_positive_rate": round(float(test_df[target_column].mean()), 4),
        },
    }

    confusion = confusion_matrix(test_df[target_column], test_pred)
    figure_path = figures_dir / f"{model_name}_confusion_matrix.png"
    display = ConfusionMatrixDisplay(confusion_matrix=confusion)
    display.plot(cmap="Blues", colorbar=False)
    plt.title(f"{model_name.upper()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    if model_name == "rf":
        artifact["feature_importance"] = save_feature_importance(
            pipeline=pipeline,
            destination_csv=metrics_dir / "rf_feature_importance.csv",
            destination_png=figures_dir / "rf_feature_importance.png",
        )

    save_json(artifact, metrics_dir / f"{model_name}_metrics.json")
    joblib.dump(pipeline, models_dir / f"{model_name}_model.joblib")

    return artifact


def save_feature_importance(
    pipeline: Pipeline,
    destination_csv: str | Path,
    destination_png: str | Path,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(destination_csv, index=False)

    top_features = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"], color="#2A6F97")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(destination_png, dpi=200)
    plt.close()

    return importance_df.head(top_n).to_dict(orient="records")
