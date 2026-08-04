#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

SPLITS = ("train", "val", "test")
INPUT_SIZE = 640
NEAR_DUPLICATE_HAMMING_THRESHOLD = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep audit of a YOLO detection dataset."
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def dhash_image(image: Image.Image) -> int:
    gray = image.convert("L").resize(
        (9, 8),
        Image.Resampling.LANCZOS,
    )

    pixels = np.asarray(gray, dtype=np.int16)

    differences = pixels[:, 1:] > pixels[:, :-1]

    value = 0

    for bit in differences.flatten():
        value = (value << 1) | int(bit)

    return value


def percentile_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
        }

    array = np.asarray(values, dtype=np.float64)

    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def load_class_names(
    dataset_root: Path,
) -> tuple[Path, list[str], dict[str, Any]]:
    yaml_files = sorted(
        list(dataset_root.rglob("*.yaml"))
        + list(dataset_root.rglob("*.yml"))
    )

    if not yaml_files:
        raise RuntimeError(
            f"No YAML file found under {dataset_root}"
        )

    for yaml_path in yaml_files:
        content = yaml.safe_load(
            yaml_path.read_text(encoding="utf-8")
        )

        if not isinstance(content, dict):
            continue

        names = content.get("names")

        if isinstance(names, list):
            return (
                yaml_path,
                [str(name) for name in names],
                content,
            )

        if isinstance(names, dict):
            normalized = {
                int(key): str(value)
                for key, value in names.items()
            }

            max_index = max(normalized)

            class_names = [
                normalized.get(index, f"UNDEFINED_{index}")
                for index in range(max_index + 1)
            ]

            return yaml_path, class_names, content

    raise RuntimeError(
        "YAML files were found, but none contains valid names."
    )


def relative_key(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix()


def add_issue(
    issues: list[dict[str, Any]],
    *,
    kind: str,
    split: str,
    image_key: str = "",
    image_path: str = "",
    label_path: str = "",
    line_number: int | str = "",
    detail: str = "",
) -> None:
    issues.append(
        {
            "kind": kind,
            "split": split,
            "image_key": image_key,
            "image_path": image_path,
            "label_path": label_path,
            "line_number": line_number,
            "detail": detail,
        }
    )


def size_category_at_640(
    box_width: float,
    box_height: float,
) -> str:
    area = box_width * box_height

    if area < 32.0 * 32.0:
        return "small"

    if area < 96.0 * 96.0:
        return "medium"

    return "large"


def render_contact_sheet(
    *,
    output_path: Path,
    title: str,
    selected_images: list[dict[str, Any]],
    boxes_by_image: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ],
    class_names: list[str],
) -> None:
    if not selected_images:
        return

    columns = 4
    rows = 3

    cell_width = 480
    cell_height = 360
    header_height = 24

    sheet = Image.new(
        "RGB",
        (
            columns * cell_width,
            rows * cell_height + 32,
        ),
        "white",
    )

    sheet_draw = ImageDraw.Draw(sheet)
    sheet_draw.text(
        (8, 8),
        title,
        fill="black",
    )

    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
    ]

    for index, record in enumerate(selected_images[:12]):
        column = index % columns
        row = index // columns

        origin_x = column * cell_width
        origin_y = 32 + row * cell_height

        cell = Image.new(
            "RGB",
            (cell_width, cell_height),
            "white",
        )

        draw = ImageDraw.Draw(cell)

        image_path = Path(record["image_path"])

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")

            original_width, original_height = image.size

            available_height = cell_height - header_height

            scale = min(
                cell_width / original_width,
                available_height / original_height,
            )

            resized_width = max(
                1,
                int(round(original_width * scale)),
            )

            resized_height = max(
                1,
                int(round(original_height * scale)),
            )

            resized = image.resize(
                (resized_width, resized_height),
                Image.Resampling.LANCZOS,
            )

            offset_x = (cell_width - resized_width) // 2
            offset_y = (
                header_height
                + (available_height - resized_height) // 2
            )

            cell.paste(
                resized,
                (offset_x, offset_y),
            )

            key = (
                record["split"],
                record["image_key"],
            )

            for box in boxes_by_image.get(key, []):
                class_id = int(box["class_id"])

                color = palette[class_id % len(palette)]

                x_center = float(box["x_center"])
                y_center = float(box["y_center"])
                box_width = float(box["box_width"])
                box_height = float(box["box_height"])

                x1 = (
                    offset_x
                    + (x_center - box_width / 2.0)
                    * resized_width
                )

                y1 = (
                    offset_y
                    + (y_center - box_height / 2.0)
                    * resized_height
                )

                x2 = (
                    offset_x
                    + (x_center + box_width / 2.0)
                    * resized_width
                )

                y2 = (
                    offset_y
                    + (y_center + box_height / 2.0)
                    * resized_height
                )

                draw.rectangle(
                    (x1, y1, x2, y2),
                    outline=color,
                    width=2,
                )

                class_name = (
                    class_names[class_id]
                    if 0 <= class_id < len(class_names)
                    else f"class_{class_id}"
                )

                label = class_name

                text_box = draw.textbbox(
                    (x1, y1),
                    label,
                )

                text_width = (
                    text_box[2] - text_box[0] + 4
                )

                text_height = (
                    text_box[3] - text_box[1] + 4
                )

                label_y = max(
                    header_height,
                    y1 - text_height,
                )

                draw.rectangle(
                    (
                        x1,
                        label_y,
                        x1 + text_width,
                        label_y + text_height,
                    ),
                    fill=color,
                )

                draw.text(
                    (x1 + 2, label_y + 2),
                    label,
                    fill="black",
                )

            header = (
                f"{record['split']} | "
                f"{record['image_key'][:42]} | "
                f"boxes={record['valid_box_count']}"
            )

            draw.text(
                (4, 4),
                header,
                fill="black",
            )

        except Exception as exc:
            draw.text(
                (8, 40),
                f"render error: {exc}",
                fill="black",
            )

        sheet.paste(
            cell,
            (origin_x, origin_y),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        quality=90,
    )


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        yaml_path,
        class_names,
        yaml_content,
    ) = load_class_names(dataset_root)

    image_records: list[dict[str, Any]] = []
    box_records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    class_counts_by_split: dict[str, Counter[int]] = {
        split: Counter()
        for split in SPLITS
    }

    split_summary: dict[str, dict[str, int]] = {}

    hash_groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    dhash_records: list[dict[str, Any]] = []

    total_label_files = 0

    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split

        image_paths = sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        label_paths = sorted(
            label_dir.rglob("*.txt")
        )

        total_label_files += len(label_paths)

        images_by_key: dict[
            str,
            list[Path],
        ] = defaultdict(list)

        labels_by_key: dict[
            str,
            list[Path],
        ] = defaultdict(list)

        for image_path in image_paths:
            images_by_key[
                relative_key(image_path, image_dir)
            ].append(image_path)

        for label_path in label_paths:
            labels_by_key[
                relative_key(label_path, label_dir)
            ].append(label_path)

        image_keys = set(images_by_key)
        label_keys = set(labels_by_key)

        for key in sorted(image_keys - label_keys):
            for image_path in images_by_key[key]:
                add_issue(
                    issues,
                    kind="image_without_label",
                    split=split,
                    image_key=key,
                    image_path=str(image_path),
                    detail="No matching TXT label file.",
                )

        for key in sorted(label_keys - image_keys):
            for label_path in labels_by_key[key]:
                add_issue(
                    issues,
                    kind="label_without_image",
                    split=split,
                    image_key=key,
                    label_path=str(label_path),
                    detail="No matching image file.",
                )

        for key, paths in images_by_key.items():
            if len(paths) > 1:
                add_issue(
                    issues,
                    kind="duplicate_image_key",
                    split=split,
                    image_key=key,
                    image_path=" | ".join(
                        str(path)
                        for path in paths
                    ),
                    detail=(
                        "Multiple image files share the same "
                        "relative stem."
                    ),
                )

        for key, paths in labels_by_key.items():
            if len(paths) > 1:
                add_issue(
                    issues,
                    kind="duplicate_label_key",
                    split=split,
                    image_key=key,
                    label_path=" | ".join(
                        str(path)
                        for path in paths
                    ),
                    detail=(
                        "Multiple label files share the same "
                        "relative stem."
                    ),
                )

        corrupt_count = 0
        empty_label_count = 0
        valid_box_count = 0
        invalid_line_count = 0

        for image_path in image_paths:
            key = relative_key(
                image_path,
                image_dir,
            )

            width = 0
            height = 0
            image_mode = ""
            image_valid = False
            image_error = ""
            image_sha256 = ""
            image_dhash = ""

            try:
                with Image.open(image_path) as image:
                    image.verify()

                with Image.open(image_path) as image:
                    width, height = image.size
                    image_mode = image.mode
                    image_dhash_value = dhash_image(image)
                    image_dhash = f"{image_dhash_value:016x}"

                image_sha256 = sha256_file(image_path)
                image_valid = True

            except Exception as exc:
                corrupt_count += 1
                image_error = (
                    f"{type(exc).__name__}: {exc}"
                )

                add_issue(
                    issues,
                    kind="corrupt_or_unreadable_image",
                    split=split,
                    image_key=key,
                    image_path=str(image_path),
                    detail=image_error,
                )

            label_candidates = labels_by_key.get(
                key,
                [],
            )

            label_path = (
                label_candidates[0]
                if label_candidates
                else None
            )

            image_valid_boxes: list[dict[str, Any]] = []
            current_invalid_lines = 0

            if label_path is not None:
                text = label_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )

                nonempty_lines = [
                    line.strip()
                    for line in text.splitlines()
                    if line.strip()
                ]

                if not nonempty_lines:
                    empty_label_count += 1

                    add_issue(
                        issues,
                        kind="empty_label_file",
                        split=split,
                        image_key=key,
                        image_path=str(image_path),
                        label_path=str(label_path),
                        detail=(
                            "Label file contains no annotations."
                        ),
                    )

                seen_annotations: set[
                    tuple[int, float, float, float, float]
                ] = set()

                for line_number, line in enumerate(
                    nonempty_lines,
                    start=1,
                ):
                    tokens = line.split()

                    if len(tokens) != 5:
                        current_invalid_lines += 1

                        add_issue(
                            issues,
                            kind="invalid_token_count",
                            split=split,
                            image_key=key,
                            image_path=str(image_path),
                            label_path=str(label_path),
                            line_number=line_number,
                            detail=(
                                f"Expected 5 tokens, got "
                                f"{len(tokens)}: {line}"
                            ),
                        )

                        continue

                    try:
                        raw_class = float(tokens[0])

                        if not raw_class.is_integer():
                            raise ValueError(
                                "class id is not an integer"
                            )

                        class_id = int(raw_class)

                        x_center = float(tokens[1])
                        y_center = float(tokens[2])
                        box_width = float(tokens[3])
                        box_height = float(tokens[4])

                    except Exception as exc:
                        current_invalid_lines += 1

                        add_issue(
                            issues,
                            kind="non_numeric_annotation",
                            split=split,
                            image_key=key,
                            image_path=str(image_path),
                            label_path=str(label_path),
                            line_number=line_number,
                            detail=(
                                f"{type(exc).__name__}: "
                                f"{exc}; line={line}"
                            ),
                        )

                        continue

                    values = (
                        x_center,
                        y_center,
                        box_width,
                        box_height,
                    )

                    if not all(
                        math.isfinite(value)
                        for value in values
                    ):
                        current_invalid_lines += 1

                        add_issue(
                            issues,
                            kind="non_finite_coordinate",
                            split=split,
                            image_key=key,
                            image_path=str(image_path),
                            label_path=str(label_path),
                            line_number=line_number,
                            detail=line,
                        )

                        continue

                    annotation = (
                        class_id,
                        x_center,
                        y_center,
                        box_width,
                        box_height,
                    )

                    if annotation in seen_annotations:
                        add_issue(
                            issues,
                            kind="duplicate_annotation",
                            split=split,
                            image_key=key,
                            image_path=str(image_path),
                            label_path=str(label_path),
                            line_number=line_number,
                            detail=line,
                        )

                    seen_annotations.add(annotation)

                    line_valid = True

                    if not (
                        0 <= class_id < len(class_names)
                    ):
                        line_valid = False

                        add_issue(
                            issues,
                            kind="class_id_out_of_range",
                            split=split,
                            image_key=key,
                            image_path=str(image_path),
                            label_path=str(label_path),
                            line_number=line_number,
                            detail=(
                                f"class_id={class_id}, "
                                f"class_count={len(class_names)}"
                            ),
                        )

                    if not (
                        0.0 <= x_center <= 1.0
                        and 0.0 <= y_center <= 1.0
                    ):
                        line_valid = False

                        add_issue(
                            issues,
                            kind="center_out_of_range",
                            split=split,
                            image_key=key,
                            image_path=str(image_path),
                            label_path=str(label_path),
                            line_number=line_number,
                            detail=line,
                        )

                    if not (
                        0.0 < box_width <= 1.0
                        and 0.0 < box_height <= 1.0
                    ):
                        line_valid = False

                        add_issue(
                            issues,
                            kind="invalid_box_size",
                            split=split,
                            image_key=key,
                            image_path=str(image_path),
                            label_path=str(label_path),
                            line_number=line_number,
                            detail=line,
                        )

                    x1 = x_center - box_width / 2.0
                    y1 = y_center - box_height / 2.0
                    x2 = x_center + box_width / 2.0
                    y2 = y_center + box_height / 2.0

                    tolerance = 1e-6

                    if (
                        x1 < -tolerance
                        or y1 < -tolerance
                        or x2 > 1.0 + tolerance
                        or y2 > 1.0 + tolerance
                    ):
                        line_valid = False

                        add_issue(
                            issues,
                            kind="box_crosses_image_boundary",
                            split=split,
                            image_key=key,
                            image_path=str(image_path),
                            label_path=str(label_path),
                            line_number=line_number,
                            detail=(
                                f"x1={x1:.8f}, y1={y1:.8f}, "
                                f"x2={x2:.8f}, y2={y2:.8f}"
                            ),
                        )

                    if not line_valid:
                        current_invalid_lines += 1
                        continue

                    pixel_width = (
                        box_width * width
                        if image_valid
                        else 0.0
                    )

                    pixel_height = (
                        box_height * height
                        if image_valid
                        else 0.0
                    )

                    letterbox_scale = (
                        min(
                            INPUT_SIZE / width,
                            INPUT_SIZE / height,
                        )
                        if image_valid
                        and width > 0
                        and height > 0
                        else 0.0
                    )

                    width_at_640 = (
                        pixel_width * letterbox_scale
                    )

                    height_at_640 = (
                        pixel_height * letterbox_scale
                    )

                    area_at_640 = (
                        width_at_640 * height_at_640
                    )

                    size_category = size_category_at_640(
                        width_at_640,
                        height_at_640,
                    )

                    box_record = {
                        "split": split,
                        "image_key": key,
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "line_number": line_number,
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "x_center": x_center,
                        "y_center": y_center,
                        "box_width": box_width,
                        "box_height": box_height,
                        "pixel_width": pixel_width,
                        "pixel_height": pixel_height,
                        "pixel_area": (
                            pixel_width * pixel_height
                        ),
                        "area_ratio": (
                            box_width * box_height
                        ),
                        "width_at_640": width_at_640,
                        "height_at_640": height_at_640,
                        "area_at_640": area_at_640,
                        "min_side_at_640": min(
                            width_at_640,
                            height_at_640,
                        ),
                        "size_category_at_640": size_category,
                    }

                    box_records.append(box_record)
                    image_valid_boxes.append(box_record)

                    class_counts_by_split[
                        split
                    ][class_id] += 1

                    valid_box_count += 1

            invalid_line_count += current_invalid_lines

            image_record = {
                "split": split,
                "image_key": key,
                "image_path": str(image_path),
                "label_path": (
                    str(label_path)
                    if label_path is not None
                    else ""
                ),
                "image_valid": image_valid,
                "image_error": image_error,
                "width": width,
                "height": height,
                "mode": image_mode,
                "sha256": image_sha256,
                "dhash": image_dhash,
                "valid_box_count": len(
                    image_valid_boxes
                ),
                "invalid_line_count": (
                    current_invalid_lines
                ),
                "class_ids": ",".join(
                    str(class_id)
                    for class_id in sorted(
                        {
                            int(box["class_id"])
                            for box in image_valid_boxes
                        }
                    )
                ),
                "min_box_area_at_640": (
                    min(
                        float(box["area_at_640"])
                        for box in image_valid_boxes
                    )
                    if image_valid_boxes
                    else ""
                ),
            }

            image_records.append(image_record)

            if image_valid:
                hash_groups[image_sha256].append(
                    image_record
                )

                dhash_records.append(
                    {
                        **image_record,
                        "dhash_int": int(
                            image_dhash,
                            16,
                        ),
                    }
                )

        split_summary[split] = {
            "image_count": len(image_paths),
            "label_file_count": len(label_paths),
            "images_without_labels": len(
                image_keys - label_keys
            ),
            "labels_without_images": len(
                label_keys - image_keys
            ),
            "corrupt_images": corrupt_count,
            "empty_label_files": empty_label_count,
            "valid_boxes": valid_box_count,
            "invalid_annotation_lines": (
                invalid_line_count
            ),
        }

    exact_duplicate_rows: list[
        dict[str, Any]
    ] = []

    exact_duplicate_group_count = 0
    cross_split_exact_group_count = 0

    for sha256_value, records in sorted(
        hash_groups.items()
    ):
        if len(records) < 2:
            continue

        exact_duplicate_group_count += 1

        splits = {
            record["split"]
            for record in records
        }

        cross_split = len(splits) > 1

        if cross_split:
            cross_split_exact_group_count += 1

        group_id = (
            f"exact_{exact_duplicate_group_count:04d}"
        )

        for record in records:
            exact_duplicate_rows.append(
                {
                    "group_id": group_id,
                    "sha256": sha256_value,
                    "group_size": len(records),
                    "cross_split": cross_split,
                    "split": record["split"],
                    "image_key": record["image_key"],
                    "image_path": record["image_path"],
                }
            )

    near_duplicate_rows: list[
        dict[str, Any]
    ] = []

    for first_index in range(
        len(dhash_records)
    ):
        first = dhash_records[first_index]

        for second_index in range(
            first_index + 1,
            len(dhash_records),
        ):
            second = dhash_records[second_index]

            if first["split"] == second["split"]:
                continue

            if first["sha256"] == second["sha256"]:
                continue

            distance = (
                int(first["dhash_int"])
                ^ int(second["dhash_int"])
            ).bit_count()

            if (
                distance
                <= NEAR_DUPLICATE_HAMMING_THRESHOLD
            ):
                near_duplicate_rows.append(
                    {
                        "hamming_distance": distance,
                        "split_a": first["split"],
                        "image_key_a": first["image_key"],
                        "image_path_a": first["image_path"],
                        "split_b": second["split"],
                        "image_key_b": second["image_key"],
                        "image_path_b": second["image_path"],
                    }
                )

    issue_counts = Counter(
        issue["kind"]
        for issue in issues
    )

    class_count_rows: list[
        dict[str, Any]
    ] = []

    global_class_counts: Counter[int] = Counter()

    for split in SPLITS:
        global_class_counts.update(
            class_counts_by_split[split]
        )

        for class_id, class_name in enumerate(
            class_names
        ):
            class_count_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "box_count": class_counts_by_split[
                        split
                    ][class_id],
                }
            )

    for class_id, class_name in enumerate(
        class_names
    ):
        class_count_rows.append(
            {
                "split": "all",
                "class_id": class_id,
                "class_name": class_name,
                "box_count": global_class_counts[
                    class_id
                ],
            }
        )

    box_size_counts = Counter(
        record["size_category_at_640"]
        for record in box_records
    )

    class_size_summary: dict[
        str,
        dict[str, Any],
    ] = {}

    for class_id, class_name in enumerate(
        class_names
    ):
        class_boxes = [
            record
            for record in box_records
            if int(record["class_id"]) == class_id
        ]

        category_counts = Counter(
            record["size_category_at_640"]
            for record in class_boxes
        )

        class_size_summary[class_name] = {
            "class_id": class_id,
            "box_count": len(class_boxes),
            "small": category_counts["small"],
            "medium": category_counts["medium"],
            "large": category_counts["large"],
            "area_at_640": percentile_summary(
                [
                    float(record["area_at_640"])
                    for record in class_boxes
                ]
            ),
            "min_side_at_640": percentile_summary(
                [
                    float(record["min_side_at_640"])
                    for record in class_boxes
                ]
            ),
        }

    image_widths = [
        float(record["width"])
        for record in image_records
        if record["image_valid"]
    ]

    image_heights = [
        float(record["height"])
        for record in image_records
        if record["image_valid"]
    ]

    box_areas_640 = [
        float(record["area_at_640"])
        for record in box_records
    ]

    box_min_sides_640 = [
        float(record["min_side_at_640"])
        for record in box_records
    ]

    total_images = len(image_records)

    total_corrupt_images = sum(
        split_summary[split]["corrupt_images"]
        for split in SPLITS
    )

    total_missing_labels = sum(
        split_summary[split][
            "images_without_labels"
        ]
        for split in SPLITS
    )

    total_orphan_labels = sum(
        split_summary[split][
            "labels_without_images"
        ]
        for split in SPLITS
    )

    total_empty_labels = sum(
        split_summary[split][
            "empty_label_files"
        ]
        for split in SPLITS
    )

    invalid_annotation_lines = sum(
        split_summary[split][
            "invalid_annotation_lines"
        ]
        for split in SPLITS
    )

    dataset_has_quality_issues = any(
        (
            total_corrupt_images > 0,
            total_missing_labels > 0,
            total_orphan_labels > 0,
            invalid_annotation_lines > 0,
            cross_split_exact_group_count > 0,
            len(near_duplicate_rows) > 0,
        )
    )

    summary = {
        "audit_execution": "PASS",
        "dataset_root": str(dataset_root),
        "dataset_yaml": str(yaml_path),
        "yaml_content": yaml_content,
        "class_names": class_names,
        "input_size_for_box_analysis": INPUT_SIZE,
        "total_images": total_images,
        "total_label_files": total_label_files,
        "total_valid_boxes": len(box_records),
        "corrupt_images": total_corrupt_images,
        "images_without_labels": total_missing_labels,
        "labels_without_images": total_orphan_labels,
        "empty_label_files": total_empty_labels,
        "invalid_annotation_lines": (
            invalid_annotation_lines
        ),
        "issue_counts": dict(issue_counts),
        "split_summary": split_summary,
        "global_class_counts": {
            class_names[class_id]: global_class_counts[
                class_id
            ]
            for class_id in range(
                len(class_names)
            )
        },
        "box_size_at_640": {
            "small": box_size_counts["small"],
            "medium": box_size_counts["medium"],
            "large": box_size_counts["large"],
        },
        "class_size_summary": class_size_summary,
        "image_width_stats": percentile_summary(
            image_widths
        ),
        "image_height_stats": percentile_summary(
            image_heights
        ),
        "box_area_at_640_stats": percentile_summary(
            box_areas_640
        ),
        "box_min_side_at_640_stats": (
            percentile_summary(
                box_min_sides_640
            )
        ),
        "exact_duplicate_groups": (
            exact_duplicate_group_count
        ),
        "cross_split_exact_duplicate_groups": (
            cross_split_exact_group_count
        ),
        "cross_split_near_duplicate_pairs": len(
            near_duplicate_rows
        ),
        "near_duplicate_hamming_threshold": (
            NEAR_DUPLICATE_HAMMING_THRESHOLD
        ),
        "dataset_has_quality_issues": (
            dataset_has_quality_issues
        ),
    }

    write_csv(
        out_dir / "image_inventory.csv",
        image_records,
        [
            "split",
            "image_key",
            "image_path",
            "label_path",
            "image_valid",
            "image_error",
            "width",
            "height",
            "mode",
            "sha256",
            "dhash",
            "valid_box_count",
            "invalid_line_count",
            "class_ids",
            "min_box_area_at_640",
        ],
    )

    write_csv(
        out_dir / "box_inventory.csv",
        box_records,
        [
            "split",
            "image_key",
            "image_path",
            "label_path",
            "line_number",
            "class_id",
            "class_name",
            "x_center",
            "y_center",
            "box_width",
            "box_height",
            "pixel_width",
            "pixel_height",
            "pixel_area",
            "area_ratio",
            "width_at_640",
            "height_at_640",
            "area_at_640",
            "min_side_at_640",
            "size_category_at_640",
        ],
    )

    write_csv(
        out_dir / "dataset_issues.csv",
        issues,
        [
            "kind",
            "split",
            "image_key",
            "image_path",
            "label_path",
            "line_number",
            "detail",
        ],
    )

    write_csv(
        out_dir / "class_counts.csv",
        class_count_rows,
        [
            "split",
            "class_id",
            "class_name",
            "box_count",
        ],
    )

    write_csv(
        out_dir / "exact_duplicate_groups.csv",
        exact_duplicate_rows,
        [
            "group_id",
            "sha256",
            "group_size",
            "cross_split",
            "split",
            "image_key",
            "image_path",
        ],
    )

    write_csv(
        out_dir / "cross_split_near_duplicates.csv",
        near_duplicate_rows,
        [
            "hamming_distance",
            "split_a",
            "image_key_a",
            "image_path_a",
            "split_b",
            "image_key_b",
            "image_path_b",
        ],
    )

    summary_json = out_dir / "summary.json"

    summary_json.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    boxes_by_image: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for box in box_records:
        boxes_by_image[
            (
                str(box["split"]),
                str(box["image_key"]),
            )
        ].append(box)

    valid_annotated_images = [
        record
        for record in image_records
        if record["image_valid"]
        and int(record["valid_box_count"]) > 0
    ]

    random_generator = random.Random(42)

    random_samples = (
        random_generator.sample(
            valid_annotated_images,
            min(
                12,
                len(valid_annotated_images),
            ),
        )
        if valid_annotated_images
        else []
    )

    crowded_samples = sorted(
        valid_annotated_images,
        key=lambda record: int(
            record["valid_box_count"]
        ),
        reverse=True,
    )[:12]

    smallest_box_samples = sorted(
        [
            record
            for record in valid_annotated_images
            if record["min_box_area_at_640"] != ""
        ],
        key=lambda record: float(
            record["min_box_area_at_640"]
        ),
    )[:12]

    render_contact_sheet(
        output_path=(
            out_dir
            / "visual_samples"
            / "random_samples.jpg"
        ),
        title="Random annotated samples",
        selected_images=random_samples,
        boxes_by_image=boxes_by_image,
        class_names=class_names,
    )

    render_contact_sheet(
        output_path=(
            out_dir
            / "visual_samples"
            / "crowded_samples.jpg"
        ),
        title="Images with the most annotations",
        selected_images=crowded_samples,
        boxes_by_image=boxes_by_image,
        class_names=class_names,
    )

    render_contact_sheet(
        output_path=(
            out_dir
            / "visual_samples"
            / "smallest_box_samples.jpg"
        ),
        title="Images containing the smallest boxes at 640",
        selected_images=smallest_box_samples,
        boxes_by_image=boxes_by_image,
        class_names=class_names,
    )

    summary_lines = [
        "============================================================",
        " Exp02.3 Raw Dataset Deep Audit Summary",
        "============================================================",
        "audit_execution=PASS",
        f"dataset_root={dataset_root}",
        f"dataset_yaml={yaml_path}",
        "class_names="
        + ",".join(
            f"{index}:{name}"
            for index, name in enumerate(
                class_names
            )
        ),
        f"total_images={total_images}",
        f"total_label_files={total_label_files}",
        f"total_valid_boxes={len(box_records)}",
        f"corrupt_images={total_corrupt_images}",
        f"images_without_labels={total_missing_labels}",
        f"labels_without_images={total_orphan_labels}",
        f"empty_label_files={total_empty_labels}",
        (
            "invalid_annotation_lines="
            f"{invalid_annotation_lines}"
        ),
        (
            "exact_duplicate_groups="
            f"{exact_duplicate_group_count}"
        ),
        (
            "cross_split_exact_duplicate_groups="
            f"{cross_split_exact_group_count}"
        ),
        (
            "cross_split_near_duplicate_pairs="
            f"{len(near_duplicate_rows)}"
        ),
        (
            "box_size_at_640="
            f"small:{box_size_counts['small']},"
            f"medium:{box_size_counts['medium']},"
            f"large:{box_size_counts['large']}"
        ),
        (
            "dataset_has_quality_issues="
            + (
                "YES"
                if dataset_has_quality_issues
                else "NO"
            )
        ),
    ]

    for split in SPLITS:
        split_data = split_summary[split]

        summary_lines.append(
            f"{split}_summary="
            f"images:{split_data['image_count']},"
            f"labels:{split_data['label_file_count']},"
            f"orphan_labels:"
            f"{split_data['labels_without_images']},"
            f"missing_labels:"
            f"{split_data['images_without_labels']},"
            f"corrupt:"
            f"{split_data['corrupt_images']},"
            f"empty:"
            f"{split_data['empty_label_files']},"
            f"boxes:"
            f"{split_data['valid_boxes']},"
            f"invalid_lines:"
            f"{split_data['invalid_annotation_lines']}"
        )

    for class_id, class_name in enumerate(
        class_names
    ):
        class_info = class_size_summary[
            class_name
        ]

        summary_lines.append(
            f"class_{class_id}_{class_name}="
            f"boxes:{class_info['box_count']},"
            f"small:{class_info['small']},"
            f"medium:{class_info['medium']},"
            f"large:{class_info['large']}"
        )

    summary_lines.extend(
        [
            f"summary_json={summary_json}",
            (
                "image_inventory="
                f"{out_dir / 'image_inventory.csv'}"
            ),
            (
                "box_inventory="
                f"{out_dir / 'box_inventory.csv'}"
            ),
            (
                "dataset_issues="
                f"{out_dir / 'dataset_issues.csv'}"
            ),
            (
                "exact_duplicates="
                f"{out_dir / 'exact_duplicate_groups.csv'}"
            ),
            (
                "near_duplicates="
                f"{out_dir / 'cross_split_near_duplicates.csv'}"
            ),
            (
                "visual_samples="
                f"{out_dir / 'visual_samples'}"
            ),
        ]
    )

    summary_text = "\n".join(
        summary_lines
    ) + "\n"

    (out_dir / "summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )

    print(summary_text, end="")


if __name__ == "__main__":
    main()
