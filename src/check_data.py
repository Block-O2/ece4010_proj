from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .dataset_builder import build_dataset
    from .utils import load_json
except ImportError:
    from dataset_builder import build_dataset
    from utils import load_json


def format_section(title: str, lines: list[str]) -> str:
    body = "\n".join(lines)
    return f"## {title}\n\n{body}\n"


def build_markdown_report(report: dict) -> str:
    lines: list[str] = ["# Austin Animal Center Data Audit", ""]

    intakes = report["intakes"]
    outcomes = report["outcomes"]
    pairing = report["pairing"]
    labels = report["labels"]

    lines.append(
        format_section(
            "Raw Files",
            [
                f"- Intakes: `{report['data_sources']['intakes']}`",
                f"- Outcomes: `{report['data_sources']['outcomes']}`",
            ],
        )
    )
    lines.append(
        format_section(
            "Intakes Summary",
            [
                f"- Rows: {intakes['rows']}",
                f"- Unique Animal IDs: {intakes['unique_animal_ids']}",
                f"- IDs with multiple records: {intakes['animal_ids_with_multiple_records']}",
                f"- Max records for one ID: {intakes['max_records_for_one_animal_id']}",
                f"- Columns: {', '.join(intakes['columns'])}",
            ],
        )
    )
    lines.append(
        format_section(
            "Outcomes Summary",
            [
                f"- Rows: {outcomes['rows']}",
                f"- Unique Animal IDs: {outcomes['unique_animal_ids']}",
                f"- IDs with multiple records: {outcomes['animal_ids_with_multiple_records']}",
                f"- Max records for one ID: {outcomes['max_records_for_one_animal_id']}",
                f"- Columns: {', '.join(outcomes['columns'])}",
            ],
        )
    )
    lines.append(
        format_section(
            "Pairing Rule",
            [
                f"- Strategy: {pairing['pairing_strategy']}",
                f"- Paired rows: {pairing['paired_rows']}",
                f"- Rows with matched outcome: {pairing['rows_with_matched_outcome']}",
                f"- Outcome match rate: {pairing['paired_outcome_rate']}",
                f"- Temporal mismatch rows: {pairing['temporal_mismatch_rows']}",
                f"- Temporal mismatch rate: {pairing['temporal_mismatch_rate']}",
            ],
        )
    )
    lines.append(
        format_section(
            "Label Summary",
            [
                f"- fast_adoption_30d positive rate: {labels['fast_adoption_30d_positive_rate']}",
                f"- adoption positive rate: {labels['adoption_positive_rate']}",
                f"- Rows with any outcome record: {labels['rows_with_any_outcome_record']}",
                f"- Rows without outcome record: {labels['rows_without_outcome_record']}",
            ],
        )
    )
    lines.append(
        format_section(
            "Fallback Note",
            [
                "- Default target remains `fast_adoption_30d`.",
                "- If repeated-ID pairing or timestamp quality makes the 30-day label unstable, switch to `adoption` without changing the rest of the pipeline.",
            ],
        )
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw Austin Animal Center data and save a markdown audit.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="data/processed/data_audit.md")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    build_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        download=not args.no_download,
    )
    report = load_json(Path(args.processed_dir) / "dataset_report.json")
    markdown = build_markdown_report(report)
    Path(args.output).write_text(markdown, encoding="utf-8")
    print(f"Saved audit report to {args.output}")


if __name__ == "__main__":
    main()
