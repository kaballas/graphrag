# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utility helpers for breaking a CSV file into smaller text snippets."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def export_csv_to_txt(
    csv_file_path: Path,
    output_dir: Path,
    *,
    records_per_file: int = 10,
) -> None:
    """Split a CSV file into text chunks to simplify manual inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file_path, sep=";", on_bad_lines="skip")
    total_records = len(df)

    for start in range(0, total_records, records_per_file):
        end = start + records_per_file
        chunk = df.iloc[start:end]

        txt_filename = f"xrecords_{start + 1}_to_{min(end, total_records)}.txt"
        txt_file_path = output_dir / txt_filename

        with txt_file_path.open("w", encoding="utf-8") as txtfile:
            for i, (_, row) in enumerate(chunk.iterrows(), start=start + 1):
                txtfile.write(f"--- Record {i} ---\n")
                for col, val in row.items():
                    txtfile.write(f"{col}: {val}\n")
                txtfile.write("\n")

        logger.info("Created file %s", txt_filename)

    logger.info("Export completed. Files saved to %s", output_dir)


def main() -> None:
    """Run the export workflow using the repository's default SAP sample data."""
    repo_root = Path(__file__).resolve().parents[1]
    default_csv = repo_root / "sap" / "input" / "123.csv"
    default_output = default_csv.parent
    export_csv_to_txt(default_csv, default_output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
