#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


REPO_DIR = Path(
    "/root/autodl-tmp/jetson-ppe-deploy-opt"
)

CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "safety_vest",
}

COLORS = {
    0: (230, 25, 75),
    1: (60, 180, 75),
    2: (0, 130, 200),
}


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
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


def newest_completed_report() -> Path:
    candidates = sorted(
        (
            path
            for path in (
                REPO_DIR
                / "results"
                / "dataset_audit"
            ).glob(
                "exp02_4c_reviewed_split_*"
            )
            if (
                path
                / "reviewed_split_manifest.csv"
            ).is_file()
            and (
                path
                / "cross_split_dhash_pairs.csv"
            ).is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise SystemExit(
            "ERROR: no completed Exp02.4c report found"
        )

    return candidates[0]


def parse_numeric_id(
    path: Path,
) -> int | None:
    match = re.search(
        r"(\d+)$",
        path.stem,
    )

    if match is None:
        return None

    return int(match.group(1))


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def read_yolo_boxes(
    label_path: Path,
) -> list[
    tuple[int, float, float, float, float]
]:
    boxes = []

    if not label_path.is_file():
        return boxes

    for raw_line in label_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        line = raw_line.strip()

        if not line:
            continue

        tokens = line.split()

        if len(tokens) != 5:
            continue

        class_id = int(float(tokens[0]))

        x_center = float(tokens[1])
        y_center = float(tokens[2])
        width = float(tokens[3])
        height = float(tokens[4])

        boxes.append(
            (
                class_id,
                x_center,
                y_center,
                width,
                height,
            )
        )

    return boxes


def render_panel(
    *,
    image_path: Path,
    label_path: Path,
    split: str,
    group_id: str,
    group_size: str,
    source_split: str,
    width: int = 620,
    height: int = 420,
) -> Image.Image:
    header_height = 58

    canvas = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        original_width, original_height = image.size

        available_height = height - header_height

        scale = min(
            width / original_width,
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

        offset_x = (
            width - resized_width
        ) // 2

        offset_y = (
            header_height
            + (
                available_height
                - resized_height
            ) // 2
        )

        canvas.paste(
            resized,
            (offset_x, offset_y),
        )

        for (
            class_id,
            x_center,
            y_center,
            box_width,
            box_height,
        ) in read_yolo_boxes(label_path):
            color = COLORS.get(
                class_id,
                (0, 0, 0),
            )

            x1 = (
                offset_x
                + (
                    x_center
                    - box_width / 2
                )
                * resized_width
            )

            y1 = (
                offset_y
                + (
                    y_center
                    - box_height / 2
                )
                * resized_height
            )

            x2 = (
                offset_x
                + (
                    x_center
                    + box_width / 2
                )
                * resized_width
            )

            y2 = (
                offset_y
                + (
                    y_center
                    + box_height / 2
                )
                * resized_height
            )

            draw.rectangle(
                (x1, y1, x2, y2),
                outline=color,
                width=2,
            )

            draw.text(
                (
                    x1 + 2,
                    max(
                        header_height,
                        y1 + 2,
                    ),
                ),
                CLASS_NAMES.get(
                    class_id,
                    f"class_{class_id}",
                ),
                fill=color,
            )

        draw.text(
            (5, 4),
            (
                f"assigned={split} | "
                f"source={source_split} | "
                f"group={group_id} | "
                f"group_size={group_size}"
            ),
            fill="black",
        )

        draw.text(
            (5, 25),
            image_path.name[:90],
            fill="black",
        )

    except Exception as exc:
        draw.text(
            (8, 80),
            (
                f"ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            fill="black",
        )

    return canvas


def save_contact_sheet(
    pairs: list[Image.Image],
    output_path: Path,
    start_index: int,
) -> None:
    if not pairs:
        return

    pair_width = pairs[0].width
    pair_height = pairs[0].height

    sheet = Image.new(
        "RGB",
        (
            pair_width,
            pair_height * len(pairs),
        ),
        "white",
    )

    for index, image in enumerate(pairs):
        sheet.paste(
            image,
            (
                0,
                index * pair_height,
            ),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        quality=92,
    )


def main() -> None:
    report_dir = newest_completed_report()

    pairs_path = (
        report_dir
        / "cross_split_dhash_pairs.csv"
    )

    manifest_path = (
        report_dir
        / "reviewed_split_manifest.csv"
    )

    candidates = read_csv(pairs_path)
    manifest = read_csv(manifest_path)

    if not candidates:
        raise SystemExit(
            "ERROR: no dHash candidates found"
        )

    manifest_by_source = {
        str(
            Path(
                row["source_image"]
            ).resolve()
        ): row
        for row in manifest
    }

    timestamp = report_dir.name.replace(
        "exp02_4c_reviewed_split_",
        "",
    )

    output_dir = (
        REPO_DIR
        / "results"
        / "dataset_audit"
        / (
            "exp02_4d_dhash_review_"
            + timestamp
        )
    )

    if output_dir.exists():
        suffix = 1

        while Path(
            f"{output_dir}_{suffix}"
        ).exists():
            suffix += 1

        output_dir = Path(
            f"{output_dir}_{suffix}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    pair_rows: list[dict[str, Any]] = []
    rendered_pairs: list[Image.Image] = []

    report_lines = [
        "Exp02.4d Cross-Split dHash Candidate Review",
        "=" * 100,
        f"source_report={report_dir}",
        f"candidate_count={len(candidates)}",
        "",
    ]

    for pair_index, pair in enumerate(
        candidates,
        start=1,
    ):
        source_a = str(
            Path(
                pair["image_a"]
            ).resolve()
        )

        source_b = str(
            Path(
                pair["image_b"]
            ).resolve()
        )

        row_a = manifest_by_source.get(
            source_a
        )

        row_b = manifest_by_source.get(
            source_b
        )

        if row_a is None or row_b is None:
            raise RuntimeError(
                "Candidate source path missing "
                "from split manifest:\n"
                f"{source_a}\n{source_b}"
            )

        image_a = Path(
            row_a["canonical_image"]
        )

        image_b = Path(
            row_b["canonical_image"]
        )

        label_a = Path(
            row_a["canonical_label"]
        )

        label_b = Path(
            row_b["canonical_label"]
        )

        numeric_a = parse_numeric_id(
            Path(source_a)
        )

        numeric_b = parse_numeric_id(
            Path(source_b)
        )

        numeric_distance = (
            abs(numeric_a - numeric_b)
            if (
                numeric_a is not None
                and numeric_b is not None
            )
            else ""
        )

        source_split_a = (
            Path(source_a).parent.name
        )

        source_split_b = (
            Path(source_b).parent.name
        )

        exact_duplicate = (
            sha256_file(image_a)
            == sha256_file(image_b)
        )

        panel_a = render_panel(
            image_path=image_a,
            label_path=label_a,
            split=row_a["assigned_split"],
            group_id=row_a[
                "reviewed_group_id"
            ],
            group_size=row_a["group_size"],
            source_split=source_split_a,
        )

        panel_b = render_panel(
            image_path=image_b,
            label_path=label_b,
            split=row_b["assigned_split"],
            group_id=row_b[
                "reviewed_group_id"
            ],
            group_size=row_b["group_size"],
            source_split=source_split_b,
        )

        pair_canvas = Image.new(
            "RGB",
            (1240, 460),
            "white",
        )

        pair_canvas.paste(
            panel_a,
            (0, 40),
        )

        pair_canvas.paste(
            panel_b,
            (620, 40),
        )

        pair_draw = ImageDraw.Draw(
            pair_canvas
        )

        pair_draw.text(
            (7, 7),
            (
                f"pair={pair_index:02d} | "
                f"dHash={pair['hamming_distance']} | "
                f"numeric_distance={numeric_distance} | "
                f"exact_duplicate={exact_duplicate}"
            ),
            fill="black",
        )

        individual_path = (
            output_dir
            / (
                f"pair_{pair_index:02d}.jpg"
            )
        )

        pair_canvas.save(
            individual_path,
            quality=94,
        )

        rendered_pairs.append(
            pair_canvas
        )

        record = {
            "pair_index": pair_index,
            "hamming_distance": pair[
                "hamming_distance"
            ],
            "numeric_id_a": (
                numeric_a
                if numeric_a is not None
                else ""
            ),
            "numeric_id_b": (
                numeric_b
                if numeric_b is not None
                else ""
            ),
            "numeric_distance": (
                numeric_distance
            ),
            "exact_duplicate": (
                exact_duplicate
            ),
            "assigned_split_a": row_a[
                "assigned_split"
            ],
            "assigned_split_b": row_b[
                "assigned_split"
            ],
            "reviewed_group_a": row_a[
                "reviewed_group_id"
            ],
            "reviewed_group_b": row_b[
                "reviewed_group_id"
            ],
            "group_size_a": row_a[
                "group_size"
            ],
            "group_size_b": row_b[
                "group_size"
            ],
            "source_split_a": (
                source_split_a
            ),
            "source_split_b": (
                source_split_b
            ),
            "source_image_a": source_a,
            "source_image_b": source_b,
            "rendered_pair": str(
                individual_path
            ),
        }

        pair_rows.append(record)

        report_lines.extend(
            [
                (
                    f"[Pair {pair_index:02d}] "
                    f"dHash="
                    f"{pair['hamming_distance']} "
                    f"numeric_distance="
                    f"{numeric_distance} "
                    f"exact_duplicate="
                    f"{exact_duplicate}"
                ),
                (
                    f"A: "
                    f"{row_a['assigned_split']} | "
                    f"{row_a['reviewed_group_id']} | "
                    f"{source_a}"
                ),
                (
                    f"B: "
                    f"{row_b['assigned_split']} | "
                    f"{row_b['reviewed_group_id']} | "
                    f"{source_b}"
                ),
                "",
            ]
        )

    write_csv(
        output_dir
        / "candidate_index.csv",
        pair_rows,
        [
            "pair_index",
            "hamming_distance",
            "numeric_id_a",
            "numeric_id_b",
            "numeric_distance",
            "exact_duplicate",
            "assigned_split_a",
            "assigned_split_b",
            "reviewed_group_a",
            "reviewed_group_b",
            "group_size_a",
            "group_size_b",
            "source_split_a",
            "source_split_b",
            "source_image_a",
            "source_image_b",
            "rendered_pair",
        ],
    )

    (
        output_dir
        / "review_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    chunk_size = 4

    for start in range(
        0,
        len(rendered_pairs),
        chunk_size,
    ):
        chunk = rendered_pairs[
            start:start + chunk_size
        ]

        end = start + len(chunk)

        save_contact_sheet(
            chunk,
            (
                output_dir
                / (
                    "candidate_pairs_"
                    f"{start + 1:02d}_"
                    f"{end:02d}.jpg"
                )
            ),
            start + 1,
        )

    archive_path = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir,
    )

    print(
        "============================================================"
    )
    print(
        " Exp02.4d dHash Candidate Review Summary"
    )
    print(
        "============================================================"
    )
    print("review_execution=PASS")
    print(f"source_report={report_dir}")
    print(
        f"candidate_count={len(candidates)}"
    )
    print(f"review_dir={output_dir}")
    print(
        "candidate_index="
        f"{output_dir / 'candidate_index.csv'}"
    )
    print(
        "review_report="
        f"{output_dir / 'review_report.txt'}"
    )
    print(f"review_archive={archive_path}")
    print(
        "exp02_4d_command_completed=YES"
    )


if __name__ == "__main__":
    main()
