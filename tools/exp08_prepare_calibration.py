#!/usr/bin/env python3
"""Build a deterministic, train-only TensorRT INT8 calibration artifact."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import random
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "person", 1: "helmet", 2: "safety_vest"}
SIZE_NAMES = ("tiny", "small", "medium", "large")


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    label_path: Path
    class_counts: Counter[str]
    size_counts: Counter[str]
    joint_counts: Counter[str]
    instance_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-count", type=int, default=1024)
    parser.add_argument("--max-proportion-delta", type=float, default=0.02)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def size_group(area_ratio: float) -> str:
    if area_ratio < 0.0025:
        return "tiny"
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.04:
        return "medium"
    return "large"


def load_record(image_path: Path, label_path: Path) -> ImageRecord:
    if not label_path.is_file():
        raise FileNotFoundError(f"missing label for {image_path.name}: {label_path}")
    class_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    joint_counts: Counter[str] = Counter()
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
        if class_id not in CLASS_NAMES:
            raise ValueError(f"{label_path}:{line_number}: invalid class {class_id}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{label_path}:{line_number}: non-finite coordinate")
        x_center, y_center, width, height = values
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            raise ValueError(f"{label_path}:{line_number}: center outside [0, 1]")
        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{label_path}:{line_number}: invalid width/height")
        class_name = CLASS_NAMES[class_id]
        group = size_group(width * height)
        class_counts[class_name] += 1
        size_counts[group] += 1
        joint_counts[f"{class_name}/{group}"] += 1
    return ImageRecord(
        image_path=image_path,
        label_path=label_path,
        class_counts=class_counts,
        size_counts=size_counts,
        joint_counts=joint_counts,
        instance_count=sum(class_counts.values()),
    )


def proportions(counter: Counter[str], keys: Iterable[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {key: 0.0 for key in keys}
    return {key: counter[key] / total for key in keys}


def distribution(records: list[ImageRecord]) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    joint_counts: Counter[str] = Counter()
    for record in records:
        class_counts.update(record.class_counts)
        size_counts.update(record.size_counts)
        joint_counts.update(record.joint_counts)
    class_keys = tuple(CLASS_NAMES.values())
    joint_keys = tuple(
        f"{class_name}/{size_name}"
        for class_name in class_keys
        for size_name in SIZE_NAMES
    )
    return {
        "image_count": len(records),
        "background_image_count": sum(
            record.instance_count == 0 for record in records
        ),
        "background_image_fraction": (
            sum(record.instance_count == 0 for record in records) / len(records)
            if records
            else 0.0
        ),
        "instance_count": sum(class_counts.values()),
        "class_counts": dict(class_counts),
        "class_proportions": proportions(class_counts, class_keys),
        "size_counts": dict(size_counts),
        "size_proportions": proportions(size_counts, SIZE_NAMES),
        "class_size_counts": dict(joint_counts),
        "class_size_proportions": proportions(joint_counts, joint_keys),
    }


def distribution_deltas(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    groups = ("class_proportions", "size_proportions", "class_size_proportions")
    deltas: dict[str, Any] = {}
    flat: list[float] = []
    for group in groups:
        values = {
            key: candidate[group][key] - reference[group][key]
            for key in reference[group]
        }
        deltas[group] = values
        flat.extend(abs(value) for value in values.values())
    background_delta = (
        candidate["background_image_fraction"]
        - reference["background_image_fraction"]
    )
    deltas["background_image_fraction"] = background_delta
    flat.append(abs(background_delta))
    deltas["max_abs_proportion_delta"] = max(flat)
    deltas["sum_abs_proportion_delta"] = sum(flat)
    return deltas


def choose_sample(
    records: list[ImageRecord], sample_count: int, seed: int, candidate_count: int
) -> tuple[list[ImageRecord], int, dict[str, Any], dict[str, Any]]:
    reference = distribution(records)
    best: tuple[tuple[float, float, int], list[ImageRecord], int, dict[str, Any], dict[str, Any]] | None = None
    for candidate_index in range(candidate_count):
        candidate_seed = seed + candidate_index
        selected = random.Random(candidate_seed).sample(records, sample_count)
        candidate_distribution = distribution(selected)
        deltas = distribution_deltas(reference, candidate_distribution)
        score = (
            deltas["max_abs_proportion_delta"],
            deltas["sum_abs_proportion_delta"],
            candidate_index,
        )
        if best is None or score < best[0]:
            best = (
                score,
                selected,
                candidate_seed,
                candidate_distribution,
                deltas,
            )
    if best is None:
        raise RuntimeError("no calibration candidate generated")
    return best[1], best[2], best[3], best[4]


def write_json(path: Path, document: Any) -> None:
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    report_dir = Path(args.report_dir).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"invalid train split: {dataset_root}")
    if args.sample_count <= 0 or args.candidate_count <= 0:
        raise ValueError("sample-count and candidate-count must be positive")
    if not report_dir.is_dir():
        raise RuntimeError(f"report directory must exist: {report_dir}")
    if not artifact_dir.is_dir() or any(artifact_dir.iterdir()):
        raise RuntimeError(f"artifact directory must exist and be empty: {artifact_dir}")

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.sample_count >= len(image_paths):
        raise ValueError(
            f"sample count {args.sample_count} must be smaller than {len(image_paths)}"
        )
    records = [
        load_record(image_path, label_dir / f"{image_path.stem}.txt")
        for image_path in image_paths
    ]
    full_distribution = distribution(records)
    selected, selected_seed, selected_distribution, deltas = choose_sample(
        records,
        args.sample_count,
        args.seed,
        args.candidate_count,
    )
    selected = sorted(selected, key=lambda record: record.image_path.name)

    manifest_entries = []
    sha_lines = []
    for record in selected:
        image_sha256 = sha256_file(record.image_path)
        manifest_entries.append(
            {
                "archive_path": f"calibration/images/{record.image_path.name}",
                "source_relative_path": f"images/train/{record.image_path.name}",
                "sha256": image_sha256,
                "bytes": record.image_path.stat().st_size,
                "instance_count": record.instance_count,
                "class_counts": dict(record.class_counts),
                "size_counts": dict(record.size_counts),
            }
        )
        sha_lines.append(
            f"{image_sha256}  calibration/images/{record.image_path.name}"
        )

    manifest = {
        "schema_version": 1,
        "source_split": "train",
        "selection_method": "best_of_deterministic_seeded_random_candidates",
        "base_seed": args.seed,
        "selected_seed": selected_seed,
        "candidate_count": args.candidate_count,
        "sample_count": args.sample_count,
        "images": manifest_entries,
    }
    manifest_path = report_dir / "calibration_manifest.json"
    write_json(manifest_path, manifest)
    (report_dir / "sha256sums.txt").write_text(
        "\n".join(sha_lines) + "\n", encoding="utf-8"
    )

    archive_path = artifact_dir / "construction_ppe3_train_calibration_256.tar.gz"
    manifest_bytes = (
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with archive_path.open("wb") as raw_archive:
        with gzip.GzipFile(fileobj=raw_archive, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for record in selected:
                    image_info = tarfile.TarInfo(
                        f"calibration/images/{record.image_path.name}"
                    )
                    image_info.size = record.image_path.stat().st_size
                    image_info.mode = 0o644
                    image_info.mtime = 0
                    with record.image_path.open("rb") as image_handle:
                        archive.addfile(image_info, image_handle)
                manifest_info = tarfile.TarInfo(
                    "calibration/calibration_manifest.json"
                )
                manifest_info.size = len(manifest_bytes)
                manifest_info.mode = 0o644
                manifest_info.mtime = 0
                archive.addfile(manifest_info, io.BytesIO(manifest_bytes))

    archive_sha256 = sha256_file(archive_path)
    result = (
        deltas["max_abs_proportion_delta"] <= args.max_proportion_delta
        and selected_distribution["background_image_count"] > 0
        and all(
            selected_distribution["class_counts"].get(name, 0) > 0
            for name in CLASS_NAMES.values()
        )
        and all(
            selected_distribution["size_counts"].get(name, 0) > 0
            for name in SIZE_NAMES
        )
    )
    summary = {
        "experiment": "Exp08.0 train-only INT8 calibration set preparation",
        "result": "PASS" if result else "FAIL",
        "dataset_root": str(dataset_root),
        "source_split": "train",
        "dataset_image_count": len(records),
        "sample_count": args.sample_count,
        "base_seed": args.seed,
        "selected_seed": selected_seed,
        "candidate_count": args.candidate_count,
        "max_proportion_delta_threshold": args.max_proportion_delta,
        "full_distribution": full_distribution,
        "selected_distribution": selected_distribution,
        "distribution_deltas": deltas,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "test_split_used": False,
    }
    write_json(report_dir / "summary.json", summary)
    (report_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Exp08.0 INT8 校准集准备",
                "",
                f"- result: `{summary['result']}`",
                "- source split: `train`（未使用 val/test）",
                f"- dataset images: {len(records)}",
                f"- selected images: {args.sample_count}",
                f"- base seed: {args.seed}",
                f"- selected seed: {selected_seed}",
                f"- candidates: {args.candidate_count}",
                f"- max proportion delta: {deltas['max_abs_proportion_delta']:.10f}",
                f"- threshold: {args.max_proportion_delta:.10f}",
                f"- archive bytes: {archive_path.stat().st_size}",
                f"- archive SHA256: `{archive_sha256}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
