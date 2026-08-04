#!/usr/bin/env python3

from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

REPO = Path("/root/autodl-tmp/jetson-ppe-deploy-opt")
DATASET = Path(
    "/root/autodl-tmp/datasets/sources/"
    "construction-ppe_ultralytics_2025_v1"
)

CLASS_IDS_TO_REVIEW = [0, 2, 5, 6, 7]
RANDOM_SEED = 42


def newest_audit_dir() -> Path:
    candidates = sorted(
        (
            path
            for path in (
                REPO / "results" / "dataset_audit"
            ).glob("exp02_3_raw_dataset_deep_audit_*")
            if (path / "summary.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise SystemExit(
            "ERROR: no completed Exp02.3 directory found"
        )

    return candidates[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def load_class_names() -> list[str]:
    data = yaml.safe_load(
        (DATASET / "data.yaml").read_text(
            encoding="utf-8"
        )
    )

    names = data["names"]

    if isinstance(names, dict):
        return [
            str(names[index])
            for index in sorted(
                int(key) for key in names
            )
        ]

    return [str(name) for name in names]


def render_image(
    image_path: Path,
    boxes: list[dict[str, str]],
    class_names: list[str],
    width: int = 520,
    height: int = 360,
) -> Image.Image:
    canvas = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        image_width, image_height = image.size

        header_height = 28
        available_height = height - header_height

        scale = min(
            width / image_width,
            available_height / image_height,
        )

        resized_width = max(
            1,
            int(round(image_width * scale)),
        )

        resized_height = max(
            1,
            int(round(image_height * scale)),
        )

        resized = image.resize(
            (resized_width, resized_height),
            Image.Resampling.LANCZOS,
        )

        offset_x = (width - resized_width) // 2
        offset_y = (
            header_height
            + (available_height - resized_height) // 2
        )

        canvas.paste(
            resized,
            (offset_x, offset_y),
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

        for box in boxes:
            class_id = int(box["class_id"])
            color = palette[class_id % len(palette)]

            x_center = float(box["x_center"])
            y_center = float(box["y_center"])
            box_width = float(box["box_width"])
            box_height = float(box["box_height"])

            x1 = (
                offset_x
                + (x_center - box_width / 2)
                * resized_width
            )

            y1 = (
                offset_y
                + (y_center - box_height / 2)
                * resized_height
            )

            x2 = (
                offset_x
                + (x_center + box_width / 2)
                * resized_width
            )

            y2 = (
                offset_y
                + (y_center + box_height / 2)
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

            draw.text(
                (x1 + 2, max(29, y1 + 2)),
                class_name,
                fill=color,
            )

        draw.text(
            (4, 5),
            image_path.name[:75],
            fill="black",
        )

    except Exception as exc:
        draw.text(
            (8, 40),
            f"ERROR: {type(exc).__name__}: {exc}",
            fill="black",
        )

    return canvas


def save_grid(
    images: list[Image.Image],
    output: Path,
    *,
    columns: int,
) -> None:
    if not images:
        return

    cell_width, cell_height = images[0].size
    rows = (len(images) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (
            columns * cell_width,
            rows * cell_height,
        ),
        "white",
    )

    for index, image in enumerate(images):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height

        sheet.paste(image, (x, y))

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output,
        quality=92,
    )


def main() -> None:
    audit_dir = newest_audit_dir()

    timestamp = audit_dir.name.rsplit("_", 2)[-2:]

    review_dir = (
        REPO
        / "results"
        / "dataset_audit"
        / (
            "exp02_3b_targeted_review_"
            + "_".join(timestamp)
        )
    )

    if review_dir.exists():
        suffix = 1

        while Path(
            f"{review_dir}_{suffix}"
        ).exists():
            suffix += 1

        review_dir = Path(
            f"{review_dir}_{suffix}"
        )

    review_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    class_names = load_class_names()

    issues = read_csv(
        audit_dir / "dataset_issues.csv"
    )

    boxes = read_csv(
        audit_dir / "box_inventory.csv"
    )

    exact_duplicates = read_csv(
        audit_dir / "exact_duplicate_groups.csv"
    )

    near_duplicates = read_csv(
        audit_dir / "cross_split_near_duplicates.csv"
    )

    boxes_by_image: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    images_by_class: dict[
        int,
        list[str],
    ] = defaultdict(list)

    for box in boxes:
        image_path = box["image_path"]

        boxes_by_image[image_path].append(box)

        class_id = int(box["class_id"])

        if image_path not in images_by_class[class_id]:
            images_by_class[class_id].append(
                image_path
            )

    missing_label_issues = [
        issue
        for issue in issues
        if issue["kind"] == "image_without_label"
    ]

    orphan_label_issues = [
        issue
        for issue in issues
        if issue["kind"] == "label_without_image"
    ]

    report_lines = [
        "Exp02.3b Targeted Dataset Review",
        "=" * 80,
        f"source_audit={audit_dir}",
        f"review_dir={review_dir}",
        "",
        "[Images without labels]",
    ]

    for issue in missing_label_issues:
        report_lines.append(
            issue["image_path"]
        )

    report_lines.extend(
        [
            "",
            "[Labels without images]",
        ]
    )

    for issue in orphan_label_issues:
        label_path = Path(issue["label_path"])

        report_lines.append(
            f"\nlabel={label_path}"
        )

        if label_path.is_file():
            content = label_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).strip()

            report_lines.append(
                content[:1000]
                if content
                else "<EMPTY>"
            )

    report_lines.extend(
        [
            "",
            "[Exact duplicate groups]",
        ]
    )

    for row in exact_duplicates:
        report_lines.append(
            f"{row['group_id']} | "
            f"cross_split={row['cross_split']} | "
            f"{row['split']} | "
            f"{row['image_path']}"
        )

    report_lines.extend(
        [
            "",
            "[Cross-split near-duplicate candidates]",
        ]
    )

    for index, row in enumerate(
        near_duplicates,
        start=1,
    ):
        report_lines.append(
            f"{index:02d} | "
            f"distance={row['hamming_distance']} | "
            f"{row['split_a']}:{row['image_key_a']} | "
            f"{row['split_b']}:{row['image_key_b']}"
        )

    (review_dir / "review_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    missing_images = []

    for issue in missing_label_issues:
        image_path = Path(issue["image_path"])

        missing_images.append(
            render_image(
                image_path,
                [],
                class_names,
            )
        )

    save_grid(
        missing_images,
        review_dir / "images_without_labels.jpg",
        columns=2,
    )

    rng = random.Random(RANDOM_SEED)

    for class_id in CLASS_IDS_TO_REVIEW:
        paths = list(images_by_class[class_id])

        rng.shuffle(paths)

        selected = paths[:12]

        rendered = [
            render_image(
                Path(image_path),
                boxes_by_image[image_path],
                class_names,
            )
            for image_path in selected
        ]

        class_name = class_names[class_id]

        save_grid(
            rendered,
            review_dir
            / (
                f"class_{class_id}_"
                f"{class_name}_samples.jpg"
            ),
            columns=3,
        )

    near_duplicate_visuals = []

    for index, row in enumerate(
        near_duplicates,
        start=1,
    ):
        first = render_image(
            Path(row["image_path_a"]),
            boxes_by_image[row["image_path_a"]],
            class_names,
        )

        second = render_image(
            Path(row["image_path_b"]),
            boxes_by_image[row["image_path_b"]],
            class_names,
        )

        pair = Image.new(
            "RGB",
            (1040, 390),
            "white",
        )

        pair.paste(first, (0, 30))
        pair.paste(second, (520, 30))

        draw = ImageDraw.Draw(pair)

        draw.text(
            (6, 6),
            (
                f"pair={index:02d} "
                f"dHash distance="
                f"{row['hamming_distance']} | "
                f"{row['split_a']} vs "
                f"{row['split_b']}"
            ),
            fill="black",
        )

        near_duplicate_visuals.append(pair)

    for start in range(
        0,
        len(near_duplicate_visuals),
        5,
    ):
        chunk = near_duplicate_visuals[
            start:start + 5
        ]

        save_grid(
            chunk,
            review_dir
            / (
                "near_duplicate_pairs_"
                f"{start + 1:02d}_"
                f"{start + len(chunk):02d}.jpg"
            ),
            columns=1,
        )

    archive_path = shutil.make_archive(
        str(review_dir),
        "zip",
        root_dir=review_dir,
    )

    print(
        "============================================================"
    )
    print(" Exp02.3b Summary")
    print(
        "============================================================"
    )
    print("review_execution=PASS")
    print(f"source_audit={audit_dir}")
    print(f"review_dir={review_dir}")
    print(
        "images_without_labels="
        f"{len(missing_label_issues)}"
    )
    print(
        "labels_without_images="
        f"{len(orphan_label_issues)}"
    )
    print(
        "exact_duplicate_rows="
        f"{len(exact_duplicates)}"
    )
    print(
        "near_duplicate_pairs="
        f"{len(near_duplicates)}"
    )
    print(
        "reviewed_class_ids="
        + ",".join(
            str(class_id)
            for class_id in CLASS_IDS_TO_REVIEW
        )
    )
    print(f"review_archive={archive_path}")
    print("exp02_3b_command_completed=YES")


if __name__ == "__main__":
    main()
