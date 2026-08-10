#!/usr/bin/env python3
"""Compare three paired/interleaved Normal and CUDA Graph Exp18 runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


EXPECTED_DIGEST = "9f3f33459f8d086a74249a57f21f158a73ca794a2229a9e1af40a03de34e2d8a"


def load_run(path: Path, expected_mode: str) -> dict[str, object]:
    summary = json.loads((path / "app_output" / "summary.json").read_text())
    validation = json.loads((path / "validation.json").read_text())
    if summary.get("result") != "PASS" or validation.get("result") != "PASS":
        raise RuntimeError(f"run is not PASS: {path}")
    if summary.get("postprocess_mode") != expected_mode:
        raise RuntimeError(f"mode mismatch: {path}")
    if validation.get("detections_sha256") != EXPECTED_DIGEST:
        raise RuntimeError(f"detection digest mismatch: {path}")
    return {"path": str(path.resolve()), "summary": summary, "validation": validation}


def improvement(normal: float, graph: float, higher_is_better: bool) -> float:
    if higher_is_better:
        return (graph - normal) / normal * 100.0
    return (normal - graph) / normal * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("runs", type=Path, nargs=6)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Frozen order: N1,G1,G2,N2,N3,G3.
    ordered_modes = ("cub", "graph", "graph", "cub", "cub", "graph")
    runs = [load_run(path, mode) for path, mode in zip(args.runs, ordered_modes)]
    pair_indices = ((0, 1), (3, 2), (4, 5))
    rows: list[dict[str, object]] = []
    for pair, (normal_index, graph_index) in enumerate(pair_indices, start=1):
        normal = runs[normal_index]["summary"]
        graph = runs[graph_index]["summary"]
        normal_timing = normal["timings_ms"]
        graph_timing = graph["timings_ms"]
        row = {
            "pair": pair,
            "normal_dir": runs[normal_index]["path"],
            "graph_dir": runs[graph_index]["path"],
            "normal_wall_fps": normal["pipeline_wall_fps"],
            "graph_wall_fps": graph["pipeline_wall_fps"],
            "wall_fps_improvement_percent": improvement(
                float(normal["pipeline_wall_fps"]),
                float(graph["pipeline_wall_fps"]), True),
            "normal_e2e_mean_ms": normal_timing["end_to_end"]["mean"],
            "graph_e2e_mean_ms": graph_timing["end_to_end"]["mean"],
            "e2e_mean_improvement_percent": improvement(
                float(normal_timing["end_to_end"]["mean"]),
                float(graph_timing["end_to_end"]["mean"]), False),
            "normal_e2e_p95_ms": normal_timing["end_to_end"]["p95"],
            "graph_e2e_p95_ms": graph_timing["end_to_end"]["p95"],
            "e2e_p95_improvement_percent": improvement(
                float(normal_timing["end_to_end"]["p95"]),
                float(graph_timing["end_to_end"]["p95"]), False),
        }
        rows.append(row)

    wall = [float(row["wall_fps_improvement_percent"]) for row in rows]
    mean = [float(row["e2e_mean_improvement_percent"]) for row in rows]
    p95 = [float(row["e2e_p95_improvement_percent"]) for row in rows]
    wall_gate = statistics.median(wall) >= 3.0 and sum(x > 0 for x in wall) >= 2
    mean_gate = statistics.median(mean) >= 3.0 and sum(x > 0 for x in mean) >= 2
    p95_gate = all(x >= -3.0 for x in p95)
    result = {
        "result": "PASS",
        "decision": "ACCEPTED" if (wall_gate or mean_gate) and p95_gate else "REJECTED",
        "order": ["N1", "G1", "G2", "N2", "N3", "G3"],
        "correctness_gate": True,
        "performance_gates": {
            "wall_fps": wall_gate,
            "e2e_mean": mean_gate,
            "e2e_p95_no_more_than_3_percent_regression": p95_gate,
        },
        "aggregate": {
            "wall_fps_improvement_median_percent": statistics.median(wall),
            "wall_fps_favorable_pairs": sum(x > 0 for x in wall),
            "e2e_mean_improvement_median_percent": statistics.median(mean),
            "e2e_mean_favorable_pairs": sum(x > 0 for x in mean),
            "e2e_p95_improvement_median_percent": statistics.median(p95),
            "e2e_p95_favorable_pairs": sum(x > 0 for x in p95),
        },
        "pairs": rows,
        "notes": [
            "Positive percentages mean Graph is better.",
            "Dynamic-frequency paired/interleaved results are the adoption evidence.",
            "A single best run is never used to override the frozen gate.",
        ],
    }
    with (output_dir / "paired_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
