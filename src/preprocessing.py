from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


AGE_PATTERN = re.compile(r"(?P<value>\d+)\s+(?P<unit>year|years|month|months|week|weeks|day|days)")
AGE_MULTIPLIERS = {
    "day": 1,
    "days": 1,
    "week": 7,
    "weeks": 7,
    "month": 30,
    "months": 30,
    "year": 365,
    "years": 365,
}


def age_to_days(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    if not text:
        return np.nan

    match = AGE_PATTERN.fullmatch(text)
    if match is None:
        return np.nan

    amount = int(match.group("value"))
    unit = match.group("unit")
    return float(amount * AGE_MULTIPLIERS[unit])


def clean_text_category(value: Any, unknown_label: str = "Unknown") -> str:
    if pd.isna(value):
        return unknown_label

    text = str(value).strip()
    if not text:
        return unknown_label

    return re.sub(r"\s+", " ", text)


def normalize_animal_type(value: Any) -> str | None:
    cleaned = clean_text_category(value, unknown_label="Unknown")
    mapping = {
        "dog": "Dog",
        "cat": "Cat",
    }
    return mapping.get(cleaned.lower(), cleaned.title())


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Collapse infrequent categorical values into a shared `Other` bucket."""

    def __init__(
        self,
        min_frequency: float = 0.01,
        max_categories: int = 20,
        other_label: str = "Other",
        unknown_label: str = "Unknown",
    ) -> None:
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.other_label = other_label
        self.unknown_label = unknown_label

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RareCategoryGrouper":
        frame = self._to_dataframe(X)
        self.columns_ = list(frame.columns)
        self.allowed_categories_: dict[str, set[str]] = {}

        for column in self.columns_:
            series = frame[column].map(lambda value: clean_text_category(value, self.unknown_label))
            value_shares = series.value_counts(normalize=True)
            kept = value_shares[value_shares >= self.min_frequency].index.tolist()
            if self.max_categories is not None:
                kept = kept[: self.max_categories]
            if not kept and not value_shares.empty:
                kept = [value_shares.index[0]]
            self.allowed_categories_[column] = set(kept)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._to_dataframe(X).copy()
        for column in self.columns_:
            allowed = self.allowed_categories_[column]
            cleaned = frame[column].map(lambda value: clean_text_category(value, self.unknown_label))
            frame[column] = cleaned.where(cleaned.isin(allowed), self.other_label)
        return frame

    @staticmethod
    def _to_dataframe(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)
