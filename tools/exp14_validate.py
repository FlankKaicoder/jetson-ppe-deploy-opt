#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--expected-detections", type=int)
    parser.add_argument("--expected-detection-sha256")
    parser.add_argument("--expected-variant")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    app_dir = args.run_dir / "app_output"
    summary_path = app_dir / "summary.json"
    frames_path = app_dir / "frames.csv"
    detections_path = app_dir / "detections.csv"
    missing = [str(path) for path in (summary_path, frames_path, detections_path)
               if not path.is_file()]
    errors = []
    if missing:
        errors.append("missing files: " + ", ".join(missing))
        result = {"result": "FAIL", "errors": errors}
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with frames_path.open(newline="", encoding="utf-8") as stream:
            frames = list(csv.DictReader(stream))
        with detections_path.open(newline="", encoding="utf-8") as stream:
            detections = list(csv.DictReader(stream))
        if summary.get("result") != "PASS":
            errors.append("summary result is not PASS")
        if int(summary.get("processed_frames", -1)) != args.expected_frames:
            errors.append("summary processed_frames mismatch")
        if len(frames) != args.expected_frames:
            errors.append("frames.csv row count mismatch")
        indices = [int(row["frame_index"]) for row in frames]
        if indices != list(range(args.expected_frames)):
            errors.append("frame indices are not contiguous and ordered")
        if args.expected_detections is not None:
            if len(detections) != args.expected_detections:
                errors.append("detection row count mismatch")
            if int(summary.get("total_detections", -1)) != args.expected_detections:
                errors.append("summary total_detections mismatch")
        detection_digest = sha256(detections_path)
        if (args.expected_detection_sha256 and
                detection_digest != args.expected_detection_sha256):
            errors.append("detections.csv SHA256 mismatch")
        if (args.expected_variant and
                summary.get("variant") != args.expected_variant):
            errors.append("variant mismatch")
        for row in frames:
            for key, value in row.items():
                if key in {"frame_index", "detection_count", "slot_index"}:
                    continue
                if not math.isfinite(float(value)) or float(value) < 0.0:
                    errors.append(f"invalid timing {key}={value}")
                    break
        result = {
            "result": "PASS" if not errors else "FAIL",
            "run_dir": str(args.run_dir.resolve()),
            "processed_frames": len(frames),
            "detection_rows": len(detections),
            "detections_sha256": detection_digest,
            "summary": summary,
            "errors": errors,
        }

    output = args.output or args.run_dir / "validation.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
