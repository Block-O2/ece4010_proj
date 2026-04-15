from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

try:
    from .feature_engineering import MODEL_FEATURE_COLUMNS
    from .modeling import build_preprocessor, create_data_splits, get_model
    from .utils import ensure_directory, load_json, save_json
except ImportError:
    from feature_engineering import MODEL_FEATURE_COLUMNS
    from modeling import build_preprocessor, create_data_splits, get_model
    from utils import ensure_directory, load_json, save_json


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
FINAL_DIR = OUTPUTS_DIR / "final_materials"
FIGURES_DIR = FINAL_DIR / "figures"


def load_modeling_dataset() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "data" / "processed" / "modeling_dataset.csv",
        parse_dates=["intake_datetime", "outcome_datetime"],
        low_memory=False,
    )


def load_metrics() -> tuple[dict, dict, dict]:
    main = load_json(OUTPUTS_DIR / "metrics" / "model_comparison.json")
    adoption = load_json(ROOT / "outputs_adoption" / "metrics" / "model_comparison.json")
    time = load_json(ROOT / "outputs_time" / "metrics" / "model_comparison.json")
    return main, adoption, time


def metric_row(setting: str, split: str, target: str, model_name: str, artifact: dict) -> dict:
    metrics = artifact["test_metrics"]
    return {
        "setting": setting,
        "split": split,
        "target": target,
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
    }


def save_markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = [str(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join([header, separator, *rows]) + "\n", encoding="utf-8")


def build_tables(main: dict, adoption: dict, time: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_rows = [
        metric_row("main_baseline", "stratified", "fast_adoption_30d", "logreg", main["logreg"]),
        metric_row("main_baseline", "stratified", "fast_adoption_30d", "rf", main["rf"]),
    ]
    comparison_rows = main_rows + [
        metric_row("fallback_target", "stratified", "adoption", "logreg", adoption["logreg"]),
        metric_row("fallback_target", "stratified", "adoption", "rf", adoption["rf"]),
        metric_row("time_split", "time", "fast_adoption_30d", "logreg", time["logreg"]),
        metric_row("time_split", "time", "fast_adoption_30d", "rf", time["rf"]),
    ]

    main_df = pd.DataFrame(main_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    main_df.to_csv(FINAL_DIR / "main_results_table.csv", index=False)
    comparison_df.to_csv(FINAL_DIR / "comparison_results_table.csv", index=False)
    save_markdown_table(main_df, FINAL_DIR / "main_results_table.md")
    save_markdown_table(comparison_df, FINAL_DIR / "comparison_results_table.md")
    return main_df, comparison_df


def clean_axes() -> None:
    plt.tight_layout()
    sns.despine()


def generate_figures(df: pd.DataFrame, main_df: pd.DataFrame, rf_feature_importance: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x="fast_adoption_30d", palette="Set2", hue="fast_adoption_30d", legend=False)
    ax.set_title("Fast Adoption Within 30 Days")
    ax.set_xlabel("fast_adoption_30d")
    ax.set_ylabel("Count")
    clean_axes()
    plt.savefig(FIGURES_DIR / "label_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x="animal_type", palette="pastel", hue="animal_type", legend=False)
    ax.set_title("Animal Type Distribution")
    ax.set_xlabel("Animal Type")
    ax.set_ylabel("Count")
    clean_axes()
    plt.savefig(FIGURES_DIR / "animal_type_distribution.png", dpi=200)
    plt.close()

    age_df = df.copy()
    age_df["age_bin"] = pd.cut(
        age_df["age_upon_intake_days"],
        bins=[-1, 90, 365, 3 * 365, 8 * 365, 30 * 365],
        labels=["0-3m", "3-12m", "1-3y", "3-8y", "8y+"],
    )
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=age_df, x="age_bin", y="fast_adoption_30d", color="#8ecae6")
    ax.set_title("Fast Adoption Rate by Age Bin")
    ax.set_xlabel("Age Bin")
    ax.set_ylabel("Fast Adoption Rate")
    clean_axes()
    plt.savefig(FIGURES_DIR / "fast_adoption_by_age_bin.png", dpi=200)
    plt.close()

    top_intake = df["intake_type"].value_counts().index[:8]
    intake_df = df[df["intake_type"].isin(top_intake)].copy()
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=intake_df,
        y="intake_type",
        x="fast_adoption_30d",
        order=top_intake,
        color="#f4a261",
    )
    ax.set_title("Fast Adoption Rate by Intake Type")
    ax.set_xlabel("Fast Adoption Rate")
    ax.set_ylabel("Intake Type")
    clean_axes()
    plt.savefig(FIGURES_DIR / "fast_adoption_by_intake_type.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=main_df, x="model", y="roc_auc", palette="deep", hue="model", legend=False)
    for idx, value in enumerate(main_df["roc_auc"]):
        ax.text(idx, value + 0.01, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Main Baseline ROC-AUC by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("ROC-AUC")
    clean_axes()
    plt.savefig(FIGURES_DIR / "main_model_comparison.png", dpi=200)
    plt.close()

    top15 = rf_feature_importance.head(15).iloc[::-1]
    plt.figure(figsize=(9, 6))
    ax = plt.barh(top15["feature"], top15["importance"], color="#457b9d")
    plt.title("Random Forest Feature Importance (Top 15)")
    plt.xlabel("Importance")
    clean_axes()
    plt.savefig(FIGURES_DIR / "rf_feature_importance_top15.png", dpi=200)
    plt.close()

    shutil.copyfile(OUTPUTS_DIR / "figures" / "rf_confusion_matrix.png", FIGURES_DIR / "main_confusion_matrix_rf.png")
    shutil.copyfile(
        OUTPUTS_DIR / "figures" / "logreg_confusion_matrix.png",
        FIGURES_DIR / "main_confusion_matrix_logreg.png",
    )


def compute_binary_metrics(y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series | None) -> dict:
    result = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "support": int(len(y_true)),
        "positive_rate": round(float(pd.Series(y_true).mean()), 4),
    }
    if y_score is not None and len(pd.Series(y_true).unique()) > 1:
        result["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
    else:
        result["roc_auc"] = None
    return result


def run_additional_analysis(df: pd.DataFrame) -> dict:
    rf_model: Pipeline = joblib.load(OUTPUTS_DIR / "models" / "rf_model.joblib")
    splits = create_data_splits(
        df,
        target_column="fast_adoption_30d",
        seed=42,
        test_size=0.2,
        val_size=0.2,
        split_method="stratified",
    )
    test_df = splits["test"].copy()
    test_pred = rf_model.predict(test_df[MODEL_FEATURE_COLUMNS])
    test_score = rf_model.predict_proba(test_df[MODEL_FEATURE_COLUMNS])[:, 1]
    test_df["prediction"] = test_pred
    test_df["score"] = test_score

    subgroup_rows: list[dict] = []
    for animal_type, group in test_df.groupby("animal_type"):
        subgroup_rows.append(
            {
                "animal_type": animal_type,
                **compute_binary_metrics(
                    group["fast_adoption_30d"],
                    group["prediction"],
                    group["score"],
                ),
            }
        )

    descriptive = (
        df.groupby("animal_type")
        .agg(
            rows=("animal_type", "size"),
            fast_adoption_rate=("fast_adoption_30d", "mean"),
            adoption_rate=("adoption", "mean"),
            median_age_days=("age_upon_intake_days", "median"),
        )
        .reset_index()
    )
    for column in ["fast_adoption_rate", "adoption_rate", "median_age_days"]:
        descriptive[column] = descriptive[column].round(4)

    features_wo_year = [feature for feature in MODEL_FEATURE_COLUMNS if feature != "intake_year"]
    preprocessor = build_preprocessor(features_wo_year, model_name="rf")
    model = get_model("rf", seed=42)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    pipeline.fit(splits["train"][features_wo_year], splits["train"]["fast_adoption_30d"])
    ablation_pred = pipeline.predict(test_df[features_wo_year])
    ablation_score = pipeline.predict_proba(test_df[features_wo_year])[:, 1]
    ablation_metrics = compute_binary_metrics(test_df["fast_adoption_30d"], ablation_pred, ablation_score)

    baseline_metrics = load_json(OUTPUTS_DIR / "metrics" / "rf_metrics.json")["test_metrics"]
    delta = {
        "accuracy_delta": round(ablation_metrics["accuracy"] - baseline_metrics["accuracy"], 4),
        "f1_score_delta": round(ablation_metrics["f1_score"] - baseline_metrics["f1_score"], 4),
        "roc_auc_delta": round(ablation_metrics["roc_auc"] - baseline_metrics["roc_auc"], 4),
    }

    return {
        "subgroup_descriptive_by_animal_type": descriptive.to_dict(orient="records"),
        "rf_test_metrics_by_animal_type": subgroup_rows,
        "intake_year_ablation_rf": {
            "features_removed": ["intake_year"],
            "baseline_rf_test_metrics": baseline_metrics,
            "ablation_rf_test_metrics": ablation_metrics,
            "delta_vs_baseline": delta,
        },
        "rerun_notes": [
            "The full main evaluation was not rerun.",
            "A lightweight random-forest ablation without intake_year was rerun on the existing processed dataset.",
            "Subgroup model performance by animal type reused the saved main random-forest model and the deterministic stratified split.",
        ],
    }


def write_additional_analysis(analysis: dict) -> None:
    save_json(analysis, FINAL_DIR / "additional_analysis.json")

    desc_df = pd.DataFrame(analysis["subgroup_descriptive_by_animal_type"])
    perf_df = pd.DataFrame(analysis["rf_test_metrics_by_animal_type"])
    ablation = analysis["intake_year_ablation_rf"]
    md = [
        "# Additional Analysis",
        "",
        "## Subgroup Descriptive Analysis by Animal Type",
        "",
        "The processed dataset was compared across dogs and cats to show how the task differs across major subgroups.",
        "",
        table_to_markdown(desc_df),
        "",
        "## Random Forest Performance by Animal Type",
        "",
        "The saved main random-forest model was applied to the deterministic stratified test split, and performance was computed separately for dogs and cats.",
        "",
        table_to_markdown(perf_df),
        "",
        "## Intake-Year Ablation",
        "",
        "A lightweight ablation removed `intake_year` from the main random-forest feature set and retrained only that model on the existing processed dataset.",
        "",
        f"Baseline RF metrics: Accuracy `{ablation['baseline_rf_test_metrics']['accuracy']}`, "
        f"F1-score `{ablation['baseline_rf_test_metrics']['f1_score']}`, "
        f"ROC-AUC `{ablation['baseline_rf_test_metrics']['roc_auc']}`.",
        "",
        f"Ablation RF metrics without `intake_year`: Accuracy `{ablation['ablation_rf_test_metrics']['accuracy']}`, "
        f"F1-score `{ablation['ablation_rf_test_metrics']['f1_score']}`, "
        f"ROC-AUC `{ablation['ablation_rf_test_metrics']['roc_auc']}`.",
        "",
        f"Change vs baseline: Accuracy `{ablation['delta_vs_baseline']['accuracy_delta']}`, "
        f"F1-score `{ablation['delta_vs_baseline']['f1_score_delta']}`, "
        f"ROC-AUC `{ablation['delta_vs_baseline']['roc_auc_delta']}`.",
        "",
        "Interpretation:",
        "",
        "- If the drop is small, the model is not relying too heavily on year alone.",
        "- If the drop is noticeable but not catastrophic, `intake_year` is useful but not the only driver.",
        "- This ablation is intentionally lightweight and is meant to strengthen the final report rather than replace the main baseline.",
        "",
        "## Rerun Notes",
        "",
    ]
    md.extend([f"- {note}" for note in analysis["rerun_notes"]])
    (FINAL_DIR / "additional_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_figures_manifest() -> None:
    rows = [
        ("label_distribution.png", "Class balance for fast_adoption_30d.", "Dataset / label definition section.", "EDA"),
        ("animal_type_distribution.png", "Counts of dogs versus cats in the modeling dataset.", "Dataset section.", "EDA"),
        ("fast_adoption_by_age_bin.png", "Fast adoption rate across coarse age bins.", "Additional analysis or discussion.", "EDA"),
        ("fast_adoption_by_intake_type.png", "Fast adoption rate across common intake types.", "Additional analysis or features discussion.", "EDA"),
        ("main_model_comparison.png", "Main baseline ROC-AUC comparison between logistic regression and random forest.", "Main results section.", "Main result"),
        ("rf_feature_importance_top15.png", "Top random-forest feature importances from the main baseline.", "Discussion / model interpretation.", "Supporting analysis"),
        ("main_confusion_matrix_rf.png", "Main random-forest confusion matrix.", "Main results section.", "Main result"),
        ("main_confusion_matrix_logreg.png", "Main logistic-regression confusion matrix.", "Main results comparison.", "Supporting analysis"),
    ]
    df = pd.DataFrame(rows, columns=["figure", "what_it_shows", "report_use", "category"])
    (FINAL_DIR / "figures_manifest.md").write_text(
        "# Figures Manifest\n\n" + table_to_markdown(df) + "\n",
        encoding="utf-8",
    )


def table_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = [str(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def main() -> None:
    ensure_directory(FINAL_DIR)
    ensure_directory(FIGURES_DIR)

    df = load_modeling_dataset()
    main, adoption, time = load_metrics()
    main_df, comparison_df = build_tables(main, adoption, time)
    rf_feature_importance = pd.read_csv(OUTPUTS_DIR / "metrics" / "rf_feature_importance.csv")
    generate_figures(df, main_df, rf_feature_importance)
    analysis = run_additional_analysis(df)
    write_additional_analysis(analysis)
    write_figures_manifest()
    print("Generated final project materials in outputs/final_materials/")


if __name__ == "__main__":
    main()
