"""Validate a prepared NIH ChestX-ray14 labels CSV."""

from __future__ import annotations

import argparse

from baseline.nih import validate_labels_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a prepared NIH labels CSV.")
    parser.add_argument("--csv", required=True, help="Path to the prepared labels CSV")
    parser.add_argument("--data-root", required=True, help="Root directory containing the images")
    args = parser.parse_args()

    validate_labels_csv(args.csv, args.data_root)


if __name__ == "__main__":
    main()
