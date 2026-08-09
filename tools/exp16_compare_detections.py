#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


TOPOLOGY_FIELDS = ("frame_index", "detection_index", "class_id", "class_name")
BOX_FIELDS = ("x1", "y1", "x2", "y2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def topology_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in TOPOLOGY_FIELDS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--expected-detections", type=int, default=151)
    parser.add_argument("--box-max-abs", type=float, default=2.0)
    parser.add_argument("--confidence-max-abs", type=float, default=0.005)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=False)

    errors: list[str] = []
    try:
        reference = read_rows(args.reference)
        candidate = read_rows(args.candidate)
    except (OSError, ValueError) as error:
        reference, candidate = [], []
        errors.append(str(error))

    if len(reference) != args.expected_detections:
        errors.append("reference detection count mismatch")
    if len(candidate) != args.expected_detections:
        errors.append("candidate detection count mismatch")
    if len(reference) != len(candidate):
        errors.append("detection row count differs")

    reference_keys = [topology_key(row) for row in reference]
    candidate_keys = [topology_key(row) for row in candidate]
    topology_equal = reference_keys == candidate_keys
    if len(set(reference_keys)) != len(reference_keys):
        errors.append("reference contains duplicate topology keys")
    if len(set(candidate_keys)) != len(candidate_keys):
        errors.append("candidate contains duplicate topology keys")
    reference_by_key = {topology_key(row): row for row in reference}
    candidate_by_key = {topology_key(row): row for row in candidate}
    missing_keys = [key for key in reference_keys if key not in candidate_by_key]
    extra_keys = [key for key in candidate_keys if key not in reference_by_key]
    if missing_keys:
        errors.append(f"missing topology keys: {missing_keys[:20]}")
    if extra_keys:
        errors.append(f"extra topology keys: {extra_keys[:20]}")
    max_box_error = 0.0
    max_confidence_error = 0.0
    max_box_error_key: tuple[str, ...] | None = None
    max_confidence_error_key: tuple[str, ...] | None = None
    matched_keys = [key for key in reference_keys if key in candidate_by_key]
    for key in matched_keys:
        expected = reference_by_key[key]
        actual = candidate_by_key[key]
        try:
            confidence_error = abs(
                float(expected["confidence"]) - float(actual["confidence"]))
            box_errors = [abs(float(expected[field]) - float(actual[field]))
                          for field in BOX_FIELDS]
        except (KeyError, ValueError) as error:
            errors.append(f"invalid numeric field key={key}: {error}")
            continue
        values = [confidence_error, *box_errors]
        if not all(math.isfinite(value) for value in values):
            errors.append(f"non-finite comparison key={key}")
            continue
        row_box_error = max(box_errors)
        if confidence_error > max_confidence_error:
            max_confidence_error = confidence_error
            max_confidence_error_key = key
        if row_box_error > max_box_error:
            max_box_error = row_box_error
            max_box_error_key = key

    if not topology_equal:
        errors.append("detection topology/order differs")
    if max_box_error > args.box_max_abs:
        errors.append("box tolerance exceeded")
    if max_confidence_error > args.confidence_max_abs:
        errors.append("confidence tolerance exceeded")

    result = {
        "experiment": "Exp16 cross-engine semantic detection comparison",
        "result": "PASS" if not errors else "FAIL",
        "configuration": {
            "expected_detections": args.expected_detections,
            "box_max_abs_threshold_source_pixels": args.box_max_abs,
            "confidence_max_abs_threshold": args.confidence_max_abs,
        },
        "sha256": {
            "reference": sha256(args.reference) if args.reference.is_file() else "",
            "candidate": sha256(args.candidate) if args.candidate.is_file() else "",
        },
        "metrics": {
            "reference_rows": len(reference),
            "candidate_rows": len(candidate),
            "matched_rows": len(matched_keys),
            "topology_and_order_equal": topology_equal,
            "missing_topology_keys": missing_keys,
            "extra_topology_keys": extra_keys,
            "max_box_abs_error_source_pixels": max_box_error,
            "max_box_abs_error_key": max_box_error_key,
            "max_confidence_abs_error": max_confidence_error,
            "max_confidence_abs_error_key": max_confidence_error_key,
        },
        "errors": errors,
    }
    (args.report_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.report_dir / "summary.txt").write_text(
        "result={result} rows={rows} topology_equal={topology} "
        "box_max_abs={box:.9f} confidence_max_abs={confidence:.9f}\n".format(
            result=result["result"], rows=len(candidate), topology=topology_equal,
            box=max_box_error, confidence=max_confidence_error),
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
