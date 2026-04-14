from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .data_loading import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_raw_data, load_csv
    from .feature_engineering import build_modeling_dataframe
    from .preprocessing import normalize_animal_type
    from .utils import ensure_directory, normalize_columns, save_json
except ImportError:
    from data_loading import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_raw_data, load_csv
    from feature_engineering import build_modeling_dataframe
    from preprocessing import normalize_animal_type
    from utils import ensure_directory, normalize_columns, save_json


INTAKE_COLUMN_CANDIDATES = {
    "animal_id": ["animal_id", "animalid"],
    "name": ["name"],
    "intake_datetime": ["datetime", "date_time", "intake_datetime"],
    "monthyear": ["monthyear", "month_year"],
    "found_location": ["found_location", "foundlocation"],
    "intake_type": ["intake_type", "intaketype"],
    "intake_condition": ["intake_condition", "intakecondition"],
    "animal_type": ["animal_type", "animaltype"],
    "sex_upon_intake": ["sex_upon_intake", "sexuponintake"],
    "age_upon_intake": ["age_upon_intake", "ageuponintake"],
    "breed": ["breed"],
    "color": ["color", "colour"],
}

OUTCOME_COLUMN_CANDIDATES = {
    "animal_id": ["animal_id", "animalid"],
    "name_outcome": ["name"],
    "outcome_datetime": ["datetime", "date_time", "outcome_datetime"],
    "outcome_monthyear": ["monthyear", "month_year"],
    "date_of_birth": ["date_of_birth", "dateofbirth"],
    "outcome_type": ["outcome_type", "outcometype"],
    "outcome_subtype": ["outcome_subtype", "outcomesubtype"],
    "animal_type_outcome": ["animal_type", "animaltype"],
    "sex_upon_outcome": ["sex_upon_outcome", "sexuponoutcome"],
    "age_upon_outcome": ["age_upon_outcome", "ageuponoutcome"],
    "outcome_breed": ["breed"],
    "outcome_color": ["color", "colour"],
}

REQUIRED_INTAKE_COLUMNS = ["animal_id", "intake_datetime", "animal_type"]
REQUIRED_OUTCOME_COLUMNS = ["animal_id", "outcome_datetime", "outcome_type"]


def parse_datetime_series(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except TypeError:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def _rename_columns(
    df: pd.DataFrame,
    candidate_map: dict[str, list[str]],
    required_columns: list[str],
) -> pd.DataFrame:
    normalized = normalize_columns(df)
    rename_map: dict[str, str] = {}
    matched_columns: set[str] = set()

    for canonical_name, candidates in candidate_map.items():
        for candidate in candidates:
            if candidate in normalized.columns:
                rename_map[candidate] = canonical_name
                matched_columns.add(canonical_name)
                break

    standardized = normalized.rename(columns=rename_map).copy()

    missing_required = [column for column in required_columns if column not in matched_columns]
    if missing_required:
        raise ValueError(
            "Missing required columns after normalization: "
            f"{missing_required}. Available columns: {list(normalized.columns)}"
        )

    for canonical_name in candidate_map:
        if canonical_name not in standardized.columns:
            standardized[canonical_name] = pd.NA

    return standardized


def standardize_intakes(df: pd.DataFrame) -> pd.DataFrame:
    standardized = _rename_columns(df, INTAKE_COLUMN_CANDIDATES, REQUIRED_INTAKE_COLUMNS)
    standardized["intake_datetime"] = parse_datetime_series(standardized["intake_datetime"])
    standardized["animal_type"] = standardized["animal_type"].map(normalize_animal_type)
    standardized = standardized.dropna(subset=["animal_id", "intake_datetime"])
    return standardized


def standardize_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    standardized = _rename_columns(df, OUTCOME_COLUMN_CANDIDATES, REQUIRED_OUTCOME_COLUMNS)
    standardized["outcome_datetime"] = parse_datetime_series(standardized["outcome_datetime"])
    standardized["animal_type_outcome"] = standardized["animal_type_outcome"].map(normalize_animal_type)
    standardized = standardized.dropna(subset=["animal_id", "outcome_datetime"])
    return standardized


def filter_core_animals(df: pd.DataFrame, animal_type_column: str) -> pd.DataFrame:
    filtered = df[df[animal_type_column].isin(["Dog", "Cat"])].copy()
    return filtered


def summarize_table(df: pd.DataFrame, dataset_name: str, datetime_column: str) -> dict[str, Any]:
    record_counts = df["animal_id"].value_counts(dropna=False)
    return {
        "dataset": dataset_name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "unique_animal_ids": int(df["animal_id"].nunique(dropna=True)),
        "records_with_missing_animal_id": int(df["animal_id"].isna().sum()),
        "records_with_missing_datetime": int(df[datetime_column].isna().sum()),
        "animal_ids_with_multiple_records": int((record_counts > 1).sum()),
        "max_records_for_one_animal_id": int(record_counts.max()) if not record_counts.empty else 0,
    }


def pair_intakes_and_outcomes(
    intakes: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    intakes_ordered = intakes.sort_values(["animal_id", "intake_datetime"]).copy()
    outcomes_ordered = outcomes.sort_values(["animal_id", "outcome_datetime"]).copy()

    intakes_ordered["pairing_sequence"] = intakes_ordered.groupby("animal_id").cumcount()
    outcomes_ordered["pairing_sequence"] = outcomes_ordered.groupby("animal_id").cumcount()

    paired = intakes_ordered.merge(
        outcomes_ordered,
        on=["animal_id", "pairing_sequence"],
        how="left",
    )

    paired["days_to_outcome"] = (
        paired["outcome_datetime"] - paired["intake_datetime"]
    ).dt.total_seconds() / 86400.0
    paired["has_outcome_record"] = paired["outcome_datetime"].notna().astype(int)
    paired["temporal_mismatch"] = paired["days_to_outcome"].lt(0).fillna(False)

    outcome_side_columns = [
        "name_outcome",
        "outcome_datetime",
        "outcome_monthyear",
        "date_of_birth",
        "outcome_type",
        "outcome_subtype",
        "animal_type_outcome",
        "sex_upon_outcome",
        "age_upon_outcome",
        "outcome_breed",
        "outcome_color",
    ]

    paired.loc[paired["temporal_mismatch"], outcome_side_columns] = pd.NA
    paired.loc[paired["temporal_mismatch"], "days_to_outcome"] = np.nan
    paired.loc[paired["temporal_mismatch"], "has_outcome_record"] = 0

    pairing_report = {
        "pairing_strategy": (
            "Chronological pairing by Animal ID: sort each animal's intake records and outcome "
            "records by datetime, then match the k-th intake to the k-th outcome."
        ),
        "paired_rows": int(len(paired)),
        "rows_with_matched_outcome": int(paired["outcome_datetime"].notna().sum()),
        "paired_outcome_rate": round(float(paired["outcome_datetime"].notna().mean()), 4),
        "temporal_mismatch_rows": int(paired["temporal_mismatch"].sum()),
        "temporal_mismatch_rate": round(float(paired["temporal_mismatch"].mean()), 4),
    }

    return paired, pairing_report


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    labeled["adoption"] = labeled["outcome_type"].eq("Adoption").astype(int)
    labeled["fast_adoption_30d"] = (
        labeled["outcome_type"].eq("Adoption")
        & labeled["days_to_outcome"].ge(0)
        & labeled["days_to_outcome"].le(30)
    ).astype(int)
    return labeled


def build_dataset(
    raw_dir: str | Path = RAW_DATA_DIR,
    processed_dir: str | Path = PROCESSED_DATA_DIR,
    download: bool = True,
) -> dict[str, Any]:
    paths = ensure_raw_data(raw_dir=raw_dir, download=download)
    processed_path = ensure_directory(processed_dir)

    intakes_raw = load_csv(paths["intakes"])
    outcomes_raw = load_csv(paths["outcomes"])

    intakes = filter_core_animals(standardize_intakes(intakes_raw), "animal_type")
    outcomes = filter_core_animals(standardize_outcomes(outcomes_raw), "animal_type_outcome")

    intake_summary = summarize_table(intakes, "intakes", "intake_datetime")
    outcome_summary = summarize_table(outcomes, "outcomes", "outcome_datetime")

    paired, pairing_report = pair_intakes_and_outcomes(intakes, outcomes)
    labeled = add_targets(paired)
    modeling_df = build_modeling_dataframe(labeled)

    paired_path = processed_path / "paired_dataset.csv"
    modeling_path = processed_path / "modeling_dataset.csv"
    paired.to_csv(paired_path, index=False)
    modeling_df.to_csv(modeling_path, index=False)

    report = {
        "data_sources": {key: str(path) for key, path in paths.items()},
        "intakes": intake_summary,
        "outcomes": outcome_summary,
        "pairing": pairing_report,
        "labels": {
            "fast_adoption_30d_positive_rate": round(float(modeling_df["fast_adoption_30d"].mean()), 4),
            "adoption_positive_rate": round(float(modeling_df["adoption"].mean()), 4),
            "rows_with_any_outcome_record": int(modeling_df["has_outcome_record"].sum()),
            "rows_without_outcome_record": int((1 - modeling_df["has_outcome_record"]).sum()),
        },
        "label_definition": {
            "fast_adoption_30d": (
                "1 if Outcome Type == Adoption and outcome_datetime - intake_datetime <= 30 days; "
                "otherwise 0."
            ),
            "fallback_adoption": "1 if Outcome Type == Adoption; otherwise 0.",
        },
    }
    save_json(report, processed_path / "dataset_report.json")

    return {
        "report": report,
        "paired_path": paired_path,
        "modeling_path": modeling_path,
        "modeling_df": modeling_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired Austin Animal Center datasets.")
    parser.add_argument("--raw-dir", default=str(RAW_DATA_DIR))
    parser.add_argument("--processed-dir", default=str(PROCESSED_DATA_DIR))
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not attempt automatic download; require CSV files in data/raw.",
    )
    args = parser.parse_args()

    result = build_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        download=not args.no_download,
    )
    print(f"Saved paired dataset to {result['paired_path']}")
    print(f"Saved modeling dataset to {result['modeling_path']}")


if __name__ == "__main__":
    main()
