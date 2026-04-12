from __future__ import annotations

import pandas as pd

try:
    from .preprocessing import age_to_days, clean_text_category, normalize_animal_type
    from .utils import season_from_month
except ImportError:
    from preprocessing import age_to_days, clean_text_category, normalize_animal_type
    from utils import season_from_month


TARGET_COLUMNS = {
    "fast30": "fast_adoption_30d",
    "adoption": "adoption",
}

MODEL_FEATURE_COLUMNS = [
    "animal_type",
    "breed",
    "color",
    "sex_upon_intake",
    "age_upon_intake_days",
    "intake_type",
    "intake_condition",
    "intake_month",
    "intake_weekday",
    "intake_season",
    "intake_year",
]


def build_modeling_dataframe(paired_df: pd.DataFrame) -> pd.DataFrame:
    df = paired_df.copy()

    df["animal_type"] = df["animal_type"].map(normalize_animal_type)
    df["breed"] = df["breed"].map(clean_text_category)
    df["color"] = df["color"].map(clean_text_category)
    df["sex_upon_intake"] = df["sex_upon_intake"].map(clean_text_category)
    df["intake_type"] = df["intake_type"].map(clean_text_category)
    df["intake_condition"] = df["intake_condition"].map(clean_text_category)
    df["age_upon_intake_days"] = df["age_upon_intake"].map(age_to_days)

    df["intake_month"] = df["intake_datetime"].dt.month.astype("Int64")
    df["intake_weekday"] = df["intake_datetime"].dt.day_name()
    df["intake_season"] = df["intake_month"].map(season_from_month)
    df["intake_year"] = df["intake_datetime"].dt.year.astype("Int64")

    ordered_columns = [
        "animal_id",
        "pairing_sequence",
        "intake_datetime",
        "outcome_datetime",
        "outcome_type",
        "days_to_outcome",
        "has_outcome_record",
        "adoption",
        "fast_adoption_30d",
        *MODEL_FEATURE_COLUMNS,
    ]

    return df[ordered_columns].copy()
