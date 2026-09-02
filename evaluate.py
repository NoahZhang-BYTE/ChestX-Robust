"""Evaluate a frozen checkpoint on one persisted split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from baseline.evaluation import evaluate_checkpoint


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen multi-label checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Optional config override; checkpoint config is used by default.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-artifact")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    result = _json_safe(
        evaluate_checkpoint(
            args.checkpoint,
            args.config,
            split=args.split,
            threshold=args.threshold,
            threshold_path=args.threshold_artifact,
        )
    )
    encoded = json.dumps(result, indent=2, ensure_ascii=True, sort_keys=True)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
