#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


RAW_BYTES = 235200
EXPECTED_DIGEST = "9f3f33459f8d086a74249a57f21f158a73ca794a2229a9e1af40a03de34e2d8a"


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
    parser.add_argument(
        "--mode",
        choices=("baseline", "raw_pinned", "atomic", "cub", "fixed"),
        required=True,
    )
    parser.add_argument("--require-file-digest", action="store_true")
    args = parser.parse_args()
    app = args.run_dir / "app_output"
    errors = []
    try:
        summary = json.loads((app / "summary.json").read_text(encoding="utf-8"))
        with (app / "frames.csv").open(newline="", encoding="utf-8") as stream:
            frames = list(csv.DictReader(stream))
        with (app / "detections.csv").open(newline="", encoding="utf-8") as stream:
            detections = list(csv.DictReader(stream))
    except (OSError, ValueError) as error:
        errors.append(str(error))
        summary, frames, detections = {}, [], []
    digest = sha256(app / "detections.csv") if (app / "detections.csv").is_file() else ""
    if summary.get("result") != "PASS":
        errors.append("application result is not PASS")
    if summary.get("postprocess_mode") != args.mode:
        errors.append("postprocess mode mismatch")
    if int(summary.get("processed_frames", -1)) != args.expected_frames:
        errors.append("processed frame count mismatch")
    if len(frames) != args.expected_frames:
        errors.append("frames.csv row count mismatch")
    if [int(row["frame_index"]) for row in frames] != list(range(args.expected_frames)):
        errors.append("frame indices are not contiguous")
    if args.require_file_digest:
        if len(detections) != 151:
            errors.append("file detection count mismatch")
        if digest != EXPECTED_DIGEST:
            errors.append("file detection digest mismatch")
    for row in frames:
        for key, value in row.items():
            if key == "postprocess_mode":
                if value != args.mode:
                    errors.append("per-frame mode mismatch")
                continue
            number = float(value)
            if not math.isfinite(number) or number < 0:
                errors.append(f"invalid numeric field {key}={value}")
                break
    mean_bytes = float(summary.get("transfer", {}).get(
        "d2h_bytes_per_frame", {}).get("mean", math.inf))
    if args.mode in ("baseline", "raw_pinned", "fixed") and mean_bytes != RAW_BYTES:
        errors.append(f"{args.mode} D2H byte count mismatch")
    if args.mode in ("atomic", "cub") and 1.0 - mean_bytes / RAW_BYTES < 0.80:
        errors.append("candidate D2H reduction is below 80%")
    result = {
        "result": "PASS" if not errors else "FAIL",
        "mode": args.mode,
        "processed_frames": len(frames),
        "detection_rows": len(detections),
        "detections_sha256": digest,
        "mean_d2h_bytes": mean_bytes,
        "d2h_reduction": 1.0 - mean_bytes / RAW_BYTES,
        "summary": summary,
        "errors": errors,
    }
    (args.run_dir / "validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
