#!/usr/bin/env python3
"""Parse TensorRT ``trtexec --dumpProfile`` output into ranked CSV/JSON."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


PROFILE_ROW = re.compile(
    r"\[I\]\s+"
    r"(?P<total_ms>\d+(?:\.\d+)?)\s+"
    r"(?P<average_ms>\d+(?:\.\d+)?)\s+"
    r"(?P<median_ms>\d+(?:\.\d+)?)\s+"
    r"(?P<percent>\d+(?:\.\d+)?)\s+"
    r"(?P<layer>.+)$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    rows = []
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PROFILE_ROW.search(line)
        if not match or match.group("layer") == "Total":
            continue
        rows.append(
            {
                "layer": match.group("layer"),
                "total_ms": float(match.group("total_ms")),
                "average_ms": float(match.group("average_ms")),
                "median_ms": float(match.group("median_ms")),
                "percent": float(match.group("percent")),
            }
        )
    if not rows:
        raise RuntimeError("no TensorRT profile rows found")
    rows.sort(key=lambda row: row["average_ms"], reverse=True)
    total_average_ms = sum(float(row["average_ms"]) for row in rows)
    result = {
        "result": "PASS",
        "layer_count": len(rows),
        "summed_average_ms": total_average_ms,
        "top": rows[: args.top],
        "notes": [
            "This is a separate trtexec diagnostic run with host transfers disabled.",
            "Per-layer profiling changes execution characteristics and is not an end-to-end benchmark.",
        ],
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "layer_profile.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "layer_profile_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
