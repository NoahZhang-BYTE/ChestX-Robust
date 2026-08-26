"""Command-line wrapper for NIH ChestX-ray14 metadata preparation."""

from __future__ import annotations

import argparse

from baseline.nih import prepare_nih_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NIH ChestX-ray14 metadata labels CSV.")
    parser.add_argument("--metadata", required=True, help="Path to Data_Entry_2017.csv")
    parser.add_argument("--image-root", required=True, help="Root directory containing NIH images")
    parser.add_argument("--output", required=True, help="Path for the prepared labels CSV")
    parser.add_argument("--seed", type=int, default=42, help="Patient split random seed")
    parser.add_argument("--train-list", help="Official NIH train/validation image list")
    parser.add_argument("--test-list", help="Official NIH test image list")
    args = parser.parse_args()

    frame = prepare_nih_metadata(
        args.metadata,
        args.image_root,
        args.output,
        seed=args.seed,
        train_list_path=args.train_list,
        test_list_path=args.test_list,
    )
    print(f"Wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
