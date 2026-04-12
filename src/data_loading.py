from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

try:
    from .utils import ensure_directory
except ImportError:
    from utils import ensure_directory


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

DATA_SOURCES = {
    "intakes": {
        "url": "https://data.austintexas.gov/api/views/wter-evkm/rows.csv?accessType=DOWNLOAD",
        "default_filename": "austin_animal_center_intakes.csv",
        "accepted_filenames": [
            "austin_animal_center_intakes.csv",
            "intakes.csv",
            "Austin_Animal_Center_Intakes.csv",
            "aac_intakes.csv",
        ],
    },
    "outcomes": {
        "url": "https://data.austintexas.gov/api/views/9t4d-g238/rows.csv?accessType=DOWNLOAD",
        "default_filename": "austin_animal_center_outcomes.csv",
        "accepted_filenames": [
            "austin_animal_center_outcomes.csv",
            "outcomes.csv",
            "Austin_Animal_Center_Outcomes.csv",
            "aac_outcomes.csv",
        ],
    },
}


def _find_existing_file(raw_dir: Path, candidates: list[str]) -> Path | None:
    for filename in candidates:
        candidate = raw_dir / filename
        if candidate.exists() and is_valid_csv_file(candidate):
            return candidate
    return None


def is_valid_csv_file(path: str | Path) -> bool:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False

    sample = file_path.read_text(encoding="utf-8", errors="ignore")[:2048].lower()
    first_line = sample.splitlines()[0] if sample else ""

    if "<html" in sample or "403 forbidden" in sample or "access denied" in sample:
        return False
    if "," not in first_line:
        return False
    return True


def _download_csv(url: str, destination: Path, timeout: int = 120) -> Path:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    if not is_valid_csv_file(destination):
        raise requests.RequestException(
            f"Downloaded file from {url} is not a valid CSV payload. "
            "The endpoint may have blocked the request."
        )
    return destination


def ensure_raw_data(
    raw_dir: str | Path = RAW_DATA_DIR,
    download: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    raw_path = ensure_directory(raw_dir)
    resolved: dict[str, Path] = {}
    missing_messages: list[str] = []

    for dataset_key, config in DATA_SOURCES.items():
        existing = None if overwrite else _find_existing_file(raw_path, config["accepted_filenames"])
        if existing is not None:
            resolved[dataset_key] = existing
            continue

        destination = raw_path / config["default_filename"]
        if download:
            try:
                resolved[dataset_key] = _download_csv(config["url"], destination)
                continue
            except requests.RequestException as exc:
                missing_messages.append(
                    f"- {dataset_key}: automatic download failed ({exc}). "
                    f"Place the CSV at `{destination}`."
                )
        else:
            missing_messages.append(
                f"- {dataset_key}: automatic download disabled. "
                f"Place the CSV at `{destination}`."
            )

    if missing_messages:
        missing_keys = [key for key in DATA_SOURCES if key not in resolved]
        if missing_keys:
            details = "\n".join(missing_messages)
            raise FileNotFoundError(
                "Raw Austin Animal Center files are not available.\n"
                "Expected two CSV files under `data/raw/`:\n"
                "- `austin_animal_center_intakes.csv`\n"
                "- `austin_animal_center_outcomes.csv`\n"
                "Details:\n"
                f"{details}"
            )

    return resolved


def load_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    return pd.read_csv(csv_path, low_memory=False)
