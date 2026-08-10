#!/usr/bin/env python3
"""Collect full-test accuracy and combine it with the frozen mixed timing screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CANDIDATES = ("p3_classification", "classification", "dfl")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tiny_small(document: dict[str, Any]) -> dict[str, float | int]:
    rows = {item["size_group"]: item for item in document["per_size"]}
    gt = int(rows["tiny"]["gt"]) + int(rows["small"]["gt"])
    tp = int(rows["tiny"]["tp"]) + int(rows["small"]["tp"])
    return {"gt": gt, "tp": tp, "recall": tp / gt}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--performance-summary", required=True, type=Path)
    args = parser.parse_args()
    fp16 = load(args.report_dir / "backend_metrics/fp16.json")["metrics"]
    fp16_scale = tiny_small(load(args.report_dir / "fp16_scale/summary.json"))
    performance = load(args.performance_summary)
    candidates: dict[str, Any] = {}
    for name in CANDIDATES:
        metrics = load(args.report_dir / f"backend_metrics/{name}.json")["metrics"]
        scale = tiny_small(load(args.report_dir / f"{name}_scale/summary.json"))
        deltas = {
            key: float(metrics[key]) - float(fp16[key])
            for key in ("precision", "recall", "map50", "map50_95")
        }
        deltas["tiny_small_recall"] = float(scale["recall"]) - float(fp16_scale["recall"])
        accuracy_gates = {
            "map50": deltas["map50"] >= -0.015,
            "map50_95": deltas["map50_95"] >= -0.010,
            "tiny_small_recall": deltas["tiny_small_recall"] >= -0.050,
        }
        perf = performance["candidates"][name]
        candidates[name] = {
            "metrics": metrics,
            "tiny_small": scale,
            "minus_fp16": deltas,
            "accuracy_gates": accuracy_gates,
            "accuracy_result": "PASS" if all(accuracy_gates.values()) else "REJECT",
            "gpu_screen": perf,
            "pareto_eligible": all(accuracy_gates.values()) and perf["screen_latency_gate"],
        }
    summary = {
        "experiment": "Exp17 three-candidate mixed-precision accuracy-latency Pareto",
        "result": "PASS",
        "meaning": "evaluation completed; individual candidates may be rejected",
        "configuration": {"images": 219, "instances": 840, "imgsz": 640,
                          "confidence": 0.25, "nms_iou": 0.70, "match_iou": 0.50},
        "thresholds": {"map50_max_drop": 0.015, "map50_95_max_drop": 0.010,
                       "tiny_small_max_drop": 0.050, "gpu_latency_min_reduction": 0.05},
        "fp16": {"metrics": fp16, "tiny_small": fp16_scale},
        "candidates": candidates,
        "decision": "NO_MIXED_CANDIDATE_ACCEPTED"
        if not any(item["pareto_eligible"] for item in candidates.values())
        else "CANDIDATE_REQUIRES_REPEAT_BUILD_AND_FINAL_PAIRED_GATE",
    }
    (args.report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = ["result=PASS", f"decision={summary['decision']}"]
    for name, item in candidates.items():
        lines.append(
            f"{name} map50={item['metrics']['map50']:.12g} "
            f"map50_95={item['metrics']['map50_95']:.12g} "
            f"tiny_small={item['tiny_small']['recall']:.12g} "
            f"accuracy={item['accuracy_result']} "
            f"gpu_median_reduction={item['gpu_screen']['median_latency_reduction_vs_fp16']:.12g} "
            f"pareto_eligible={item['pareto_eligible']}"
        )
    lines.append("")
    (args.report_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
