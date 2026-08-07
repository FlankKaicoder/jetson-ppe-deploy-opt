#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    detections_path = run_dir / "detections.csv"
    frames_path = run_dir / "frames.csv"
    if not all(path.is_file() for path in (summary_path, detections_path, frames_path)):
        raise RuntimeError(f"missing required output in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("result") != "PASS":
        raise RuntimeError(f"non-PASS summary in {run_dir}")
    with frames_path.open(newline="", encoding="utf-8") as stream:
        frames = list(csv.DictReader(stream))
    if len(frames) != int(summary["processed_frames"]):
        raise RuntimeError(f"frame count mismatch in {run_dir}")
    with detections_path.open(newline="", encoding="utf-8") as stream:
        detections = list(csv.DictReader(stream))
    if len(detections) != int(summary["total_detections"]):
        raise RuntimeError(f"detection count mismatch in {run_dir}")
    for detection in detections:
        class_id = int(detection["class_id"])
        confidence = float(detection["confidence"])
        x1, y1, x2, y2 = (float(detection[key]) for key in ("x1", "y1", "x2", "y2"))
        if class_id not in (0, 1, 2) or not 0.25 <= confidence <= 1.0:
            raise RuntimeError(f"invalid class/confidence in {run_dir}")
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise RuntimeError(f"invalid box in {run_dir}")
    return {
        "run_dir": str(run_dir),
        "source_type": summary["source_type"],
        "processed_frames": int(summary["processed_frames"]),
        "total_detections": int(summary["total_detections"]),
        "detections_sha256": sha256(detections_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audits = [audit(path.resolve()) for path in args.run_dirs]
    if len(audits) > 1:
        signatures = {
            (item["processed_frames"], item["total_detections"], item["detections_sha256"])
            for item in audits
        }
        if len(signatures) != 1:
            raise RuntimeError("formal file runs are not deterministic")
    result = {"result": "PASS", "runs": audits}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
