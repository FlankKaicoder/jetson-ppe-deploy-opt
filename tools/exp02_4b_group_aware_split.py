#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

TARGET_NAMES = {
    0: "person",
    1: "helmet",
    2: "safety_vest",
}

SPLITS = ("train", "val", "test")

TARGET_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

NUMERIC_BLOCK_SIZE = 20
GLOBAL_DHASH_THRESHOLD = 4
BOUNDARY_DHASH_THRESHOLD = 12
BOUNDARY_NUMERIC_DISTANCE = 3
RANDOM_SEED = 42


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[
                self.parent[value]
            ]
            value = self.parent[value]

        return value

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return

        if (
            self.rank[root_first]
            < self.rank[root_second]
        ):
            root_first, root_second = (
                root_second,
                root_first,
            )

        self.parent[root_second] = root_first

        if (
            self.rank[root_first]
            == self.rank[root_second]
        ):
            self.rank[root_first] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--canonical-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--canonical-manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--dataset-out",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--report-out",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
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
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def dhash_image(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize(
            (9, 8),
            Image.Resampling.LANCZOS,
        )

    pixels = np.asarray(
        gray,
        dtype=np.int16,
    )

    differences = (
        pixels[:, 1:]
        > pixels[:, :-1]
    )

    value = 0

    for bit in differences.flatten():
        value = (value << 1) | int(bit)

    return value


def parse_trailing_number(path: Path) -> int | None:
    match = re.search(
        r"(\d+)$",
        path.stem,
    )

    if match is None:
        return None

    return int(match.group(1))


def count_classes(
    label_path: Path,
) -> Counter[int]:
    counts: Counter[int] = Counter()

    if not label_path.is_file():
        raise RuntimeError(
            f"Label missing: {label_path}"
        )

    for line_number, raw_line in enumerate(
        label_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        tokens = line.split()

        if len(tokens) != 5:
            raise RuntimeError(
                f"{label_path}:{line_number}: "
                f"expected 5 tokens"
            )

        class_id = int(float(tokens[0]))

        if class_id not in TARGET_NAMES:
            raise RuntimeError(
                f"{label_path}:{line_number}: "
                f"unexpected class id {class_id}"
            )

        counts[class_id] += 1

    return counts


def hardlink_or_copy(
    source: Path,
    destination: Path,
) -> str:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        os.link(source, destination)
        return "hardlink"

    except OSError:
        shutil.copy2(
            source,
            destination,
        )
        return "copy"


def assignment_cost(
    *,
    candidate_split: str,
    group_image_count: int,
    group_class_counts: Counter[int],
    group_negative_count: int,
    current_images: dict[str, int],
    current_classes: dict[str, Counter[int]],
    current_negatives: dict[str, int],
    target_images: dict[str, float],
    target_classes: dict[str, dict[int, float]],
    target_negatives: dict[str, float],
) -> float:
    simulated_images = dict(current_images)

    simulated_classes = {
        split: Counter(counter)
        for split, counter
        in current_classes.items()
    }

    simulated_negatives = dict(
        current_negatives
    )

    simulated_images[
        candidate_split
    ] += group_image_count

    simulated_classes[
        candidate_split
    ].update(group_class_counts)

    simulated_negatives[
        candidate_split
    ] += group_negative_count

    image_cost = 0.0
    class_cost = 0.0
    negative_cost = 0.0
    overflow_penalty = 0.0

    for split in SPLITS:
        image_target = target_images[split]

        image_cost += (
            (
                simulated_images[split]
                - image_target
            )
            / max(image_target, 1.0)
        ) ** 2

        if (
            simulated_images[split]
            > image_target * 1.10
        ):
            overflow_penalty += (
                simulated_images[split]
                - image_target * 1.10
            ) ** 2

        for class_id in TARGET_NAMES:
            class_target = target_classes[
                split
            ][class_id]

            class_cost += (
                (
                    simulated_classes[
                        split
                    ][class_id]
                    - class_target
                )
                / max(class_target, 1.0)
            ) ** 2

        negative_target = target_negatives[
            split
        ]

        negative_cost += (
            (
                simulated_negatives[split]
                - negative_target
            )
            / max(negative_target, 1.0)
        ) ** 2

    return (
        image_cost
        + 0.75 * class_cost
        + 0.15 * negative_cost
        + 0.001 * overflow_penalty
    )


def main() -> None:
    args = parse_args()

    canonical_root = (
        args.canonical_root.resolve()
    )

    canonical_manifest = (
        args.canonical_manifest.resolve()
    )

    dataset_out = (
        args.dataset_out.resolve()
    )

    report_out = (
        args.report_out.resolve()
    )

    if not canonical_root.is_dir():
        raise SystemExit(
            "ERROR: canonical dataset not found: "
            f"{canonical_root}"
        )

    if not canonical_manifest.is_file():
        raise SystemExit(
            "ERROR: canonical manifest not found: "
            f"{canonical_manifest}"
        )

    if dataset_out.exists():
        raise SystemExit(
            "ERROR: output already exists: "
            f"{dataset_out}"
        )

    report_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = read_csv(
        canonical_manifest
    )

    records: list[dict[str, Any]] = []

    for row in manifest:
        image_path = Path(
            row["canonical_image"]
        ).resolve()

        label_path = Path(
            row["canonical_label"]
        ).resolve()

        source_image = Path(
            row["source_image"]
        ).resolve()

        if not image_path.is_file():
            raise RuntimeError(
                f"Canonical image missing: "
                f"{image_path}"
            )

        if not label_path.is_file():
            raise RuntimeError(
                f"Canonical label missing: "
                f"{label_path}"
            )

        numeric_id = parse_trailing_number(
            source_image
        )

        class_counts = count_classes(
            label_path
        )

        record = {
            "index": len(records),
            "canonical_image": image_path,
            "canonical_label": label_path,
            "source_image": source_image,
            "source_split": row[
                "source_split"
            ],
            "numeric_id": numeric_id,
            "numeric_block": (
                numeric_id
                // NUMERIC_BLOCK_SIZE
                if numeric_id is not None
                else None
            ),
            "class_counts": class_counts,
            "box_count": sum(
                class_counts.values()
            ),
            "is_negative": (
                sum(class_counts.values()) == 0
            ),
            "sha256": sha256_file(
                image_path
            ),
            "dhash": dhash_image(
                image_path
            ),
        }

        records.append(record)

    total_images = len(records)

    if total_images != 1416:
        raise RuntimeError(
            "Expected 1416 canonical images, "
            f"found {total_images}"
        )

    union_find = UnionFind(
        total_images
    )

    block_members: dict[
        int,
        list[int],
    ] = defaultdict(list)

    for record in records:
        numeric_block = record[
            "numeric_block"
        ]

        if numeric_block is not None:
            block_members[
                int(numeric_block)
            ].append(
                int(record["index"])
            )

    numeric_block_union_count = 0

    for indices in block_members.values():
        if len(indices) < 2:
            continue

        first_index = indices[0]

        for index in indices[1:]:
            union_find.union(
                first_index,
                index,
            )

            numeric_block_union_count += 1

    global_visual_edges = 0
    boundary_visual_edges = 0

    for first_index in range(
        total_images
    ):
        first = records[first_index]

        for second_index in range(
            first_index + 1,
            total_images,
        ):
            second = records[second_index]

            distance = (
                int(first["dhash"])
                ^ int(second["dhash"])
            ).bit_count()

            should_union = False

            if (
                distance
                <= GLOBAL_DHASH_THRESHOLD
            ):
                should_union = True
                global_visual_edges += 1

            else:
                first_number = first[
                    "numeric_id"
                ]

                second_number = second[
                    "numeric_id"
                ]

                if (
                    first_number is not None
                    and second_number is not None
                    and abs(
                        int(first_number)
                        - int(second_number)
                    )
                    <= BOUNDARY_NUMERIC_DISTANCE
                    and distance
                    <= BOUNDARY_DHASH_THRESHOLD
                ):
                    should_union = True
                    boundary_visual_edges += 1

            if should_union:
                union_find.union(
                    first_index,
                    second_index,
                )

    root_members: dict[
        int,
        list[int],
    ] = defaultdict(list)

    for index in range(total_images):
        root_members[
            union_find.find(index)
        ].append(index)

    ordered_groups = sorted(
        root_members.values(),
        key=lambda members: (
            -len(members),
            min(members),
        ),
    )

    groups: list[dict[str, Any]] = []

    for group_number, members in enumerate(
        ordered_groups,
        start=1,
    ):
        class_counts: Counter[int] = Counter()

        negative_count = 0

        for index in members:
            record = records[index]

            class_counts.update(
                record["class_counts"]
            )

            if record["is_negative"]:
                negative_count += 1

        groups.append(
            {
                "group_id": (
                    f"group_{group_number:04d}"
                ),
                "members": members,
                "image_count": len(members),
                "class_counts": class_counts,
                "negative_count": negative_count,
            }
        )

    max_group_size = max(
        group["image_count"]
        for group in groups
    )

    if max_group_size > 300:
        raise RuntimeError(
            "Grouping created an unexpectedly "
            f"large component: {max_group_size}"
        )

    global_class_counts: Counter[int] = (
        Counter()
    )

    total_negative_images = 0

    for record in records:
        global_class_counts.update(
            record["class_counts"]
        )

        if record["is_negative"]:
            total_negative_images += 1

    target_images = {
        split: (
            total_images
            * TARGET_RATIOS[split]
        )
        for split in SPLITS
    }

    target_classes = {
        split: {
            class_id: (
                global_class_counts[class_id]
                * TARGET_RATIOS[split]
            )
            for class_id in TARGET_NAMES
        }
        for split in SPLITS
    }

    target_negatives = {
        split: (
            total_negative_images
            * TARGET_RATIOS[split]
        )
        for split in SPLITS
    }

    current_images = {
        split: 0
        for split in SPLITS
    }

    current_classes = {
        split: Counter()
        for split in SPLITS
    }

    current_negatives = {
        split: 0
        for split in SPLITS
    }

    rng = random.Random(
        RANDOM_SEED
    )

    groups_for_assignment = list(groups)

    rng.shuffle(
        groups_for_assignment
    )

    groups_for_assignment.sort(
        key=lambda group: (
            -int(group["image_count"]),
            -sum(
                group["class_counts"].values()
            ),
        )
    )

    group_assignment: dict[
        str,
        str,
    ] = {}

    for group in groups_for_assignment:
        candidate_costs = []

        for split in SPLITS:
            cost = assignment_cost(
                candidate_split=split,
                group_image_count=int(
                    group["image_count"]
                ),
                group_class_counts=group[
                    "class_counts"
                ],
                group_negative_count=int(
                    group["negative_count"]
                ),
                current_images=current_images,
                current_classes=current_classes,
                current_negatives=current_negatives,
                target_images=target_images,
                target_classes=target_classes,
                target_negatives=target_negatives,
            )

            candidate_costs.append(
                (
                    cost,
                    (
                        current_images[split]
                        / max(
                            target_images[split],
                            1.0,
                        )
                    ),
                    split,
                )
            )

        _, _, selected_split = min(
            candidate_costs
        )

        group_assignment[
            group["group_id"]
        ] = selected_split

        current_images[
            selected_split
        ] += int(
            group["image_count"]
        )

        current_classes[
            selected_split
        ].update(
            group["class_counts"]
        )

        current_negatives[
            selected_split
        ] += int(
            group["negative_count"]
        )

    dataset_out.mkdir(
        parents=True,
        exist_ok=False,
    )

    for split in SPLITS:
        (
            dataset_out
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            dataset_out
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    group_rows: list[dict[str, Any]] = []
    transfer_counts: Counter[str] = Counter()

    record_split_assignment: dict[
        int,
        str,
    ] = {}

    for group in groups:
        split = group_assignment[
            group["group_id"]
        ]

        for index in group["members"]:
            record = records[index]

            record_split_assignment[
                index
            ] = split

            image_source = record[
                "canonical_image"
            ]

            label_source = record[
                "canonical_label"
            ]

            image_destination = (
                dataset_out
                / "images"
                / split
                / image_source.name
            )

            label_destination = (
                dataset_out
                / "labels"
                / split
                / label_source.name
            )

            image_action = hardlink_or_copy(
                image_source,
                image_destination,
            )

            label_action = hardlink_or_copy(
                label_source,
                label_destination,
            )

            transfer_counts[
                f"image_{image_action}"
            ] += 1

            transfer_counts[
                f"label_{label_action}"
            ] += 1

            class_counts = record[
                "class_counts"
            ]

            group_rows.append(
                {
                    "group_id": group[
                        "group_id"
                    ],
                    "assigned_split": split,
                    "group_size": group[
                        "image_count"
                    ],
                    "canonical_image": str(
                        image_source
                    ),
                    "canonical_label": str(
                        label_source
                    ),
                    "source_image": str(
                        record["source_image"]
                    ),
                    "source_split": record[
                        "source_split"
                    ],
                    "numeric_id": (
                        record["numeric_id"]
                        if record[
                            "numeric_id"
                        ] is not None
                        else ""
                    ),
                    "numeric_block": (
                        record["numeric_block"]
                        if record[
                            "numeric_block"
                        ] is not None
                        else ""
                    ),
                    "sha256": record["sha256"],
                    "dhash": (
                        f"{int(record['dhash']):016x}"
                    ),
                    "person_boxes": (
                        class_counts[0]
                    ),
                    "helmet_boxes": (
                        class_counts[1]
                    ),
                    "safety_vest_boxes": (
                        class_counts[2]
                    ),
                    "is_negative": record[
                        "is_negative"
                    ],
                    "image_transfer": (
                        image_action
                    ),
                    "label_transfer": (
                        label_action
                    ),
                }
            )

    cross_split_close_pairs = []

    for first_index in range(
        total_images
    ):
        first_split = (
            record_split_assignment[
                first_index
            ]
        )

        for second_index in range(
            first_index + 1,
            total_images,
        ):
            second_split = (
                record_split_assignment[
                    second_index
                ]
            )

            if first_split == second_split:
                continue

            distance = (
                int(records[first_index]["dhash"])
                ^ int(records[second_index]["dhash"])
            ).bit_count()

            if (
                distance
                <= GLOBAL_DHASH_THRESHOLD
            ):
                cross_split_close_pairs.append(
                    {
                        "hamming_distance": distance,
                        "split_a": first_split,
                        "image_a": str(
                            records[
                                first_index
                            ][
                                "canonical_image"
                            ]
                        ),
                        "split_b": second_split,
                        "image_b": str(
                            records[
                                second_index
                            ][
                                "canonical_image"
                            ]
                        ),
                    }
                )

    split_summary: dict[
        str,
        dict[str, Any],
    ] = {}

    for split in SPLITS:
        image_count = len(
            list(
                (
                    dataset_out
                    / "images"
                    / split
                ).glob("*")
            )
        )

        label_count = len(
            list(
                (
                    dataset_out
                    / "labels"
                    / split
                ).glob("*.txt")
            )
        )

        split_summary[split] = {
            "images": image_count,
            "labels": label_count,
            "ratio": (
                image_count / total_images
            ),
            "negative_images": (
                current_negatives[split]
            ),
            "class_counts": {
                TARGET_NAMES[
                    class_id
                ]: current_classes[
                    split
                ][class_id]
                for class_id in TARGET_NAMES
            },
        }

    total_output_images = sum(
        split_summary[split]["images"]
        for split in SPLITS
    )

    total_output_labels = sum(
        split_summary[split]["labels"]
        for split in SPLITS
    )

    if total_output_images != total_images:
        raise RuntimeError(
            "Output image total mismatch: "
            f"{total_output_images} "
            f"!= {total_images}"
        )

    if total_output_labels != total_images:
        raise RuntimeError(
            "Output label total mismatch: "
            f"{total_output_labels} "
            f"!= {total_images}"
        )

    if cross_split_close_pairs:
        raise RuntimeError(
            "Detected cross-split dHash pairs "
            f"within threshold: "
            f"{len(cross_split_close_pairs)}"
        )

    dataset_yaml = {
        "path": str(dataset_out),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": TARGET_NAMES,
    }

    (
        dataset_out
        / "construction_ppe3.yaml"
    ).write_text(
        yaml.safe_dump(
            dataset_yaml,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    write_csv(
        report_out
        / "group_split_manifest.csv",
        group_rows,
        [
            "group_id",
            "assigned_split",
            "group_size",
            "canonical_image",
            "canonical_label",
            "source_image",
            "source_split",
            "numeric_id",
            "numeric_block",
            "sha256",
            "dhash",
            "person_boxes",
            "helmet_boxes",
            "safety_vest_boxes",
            "is_negative",
            "image_transfer",
            "label_transfer",
        ],
    )

    write_csv(
        report_out
        / "cross_split_close_pairs.csv",
        cross_split_close_pairs,
        [
            "hamming_distance",
            "split_a",
            "image_a",
            "split_b",
            "image_b",
        ],
    )

    metadata = {
        "split_execution": "PASS",
        "canonical_root": str(
            canonical_root
        ),
        "canonical_manifest": str(
            canonical_manifest
        ),
        "dataset_out": str(
            dataset_out
        ),
        "seed": RANDOM_SEED,
        "target_ratios": TARGET_RATIOS,
        "numeric_block_size": (
            NUMERIC_BLOCK_SIZE
        ),
        "global_dhash_threshold": (
            GLOBAL_DHASH_THRESHOLD
        ),
        "boundary_dhash_threshold": (
            BOUNDARY_DHASH_THRESHOLD
        ),
        "boundary_numeric_distance": (
            BOUNDARY_NUMERIC_DISTANCE
        ),
        "total_images": total_images,
        "group_count": len(groups),
        "max_group_size": max_group_size,
        "numeric_block_union_count": (
            numeric_block_union_count
        ),
        "global_visual_edges": (
            global_visual_edges
        ),
        "boundary_visual_edges": (
            boundary_visual_edges
        ),
        "global_class_counts": {
            TARGET_NAMES[class_id]: (
                global_class_counts[
                    class_id
                ]
            )
            for class_id in TARGET_NAMES
        },
        "total_negative_images": (
            total_negative_images
        ),
        "split_summary": split_summary,
        "cross_split_dhash_pairs": len(
            cross_split_close_pairs
        ),
        "transfer_counts": dict(
            transfer_counts
        ),
    }

    (
        report_out
        / "summary.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        dataset_out
        / ".exp02_4b_complete"
    ).write_text(
        "PASS\n",
        encoding="utf-8",
    )

    print(
        "============================================================"
    )
    print(
        " Exp02.4b Group-Aware Split Summary"
    )
    print(
        "============================================================"
    )
    print("split_execution=PASS")
    print(
        f"canonical_root={canonical_root}"
    )
    print(
        f"dataset_out={dataset_out}"
    )
    print(
        f"total_images={total_images}"
    )
    print(
        f"group_count={len(groups)}"
    )
    print(
        f"max_group_size={max_group_size}"
    )
    print(
        "numeric_block_union_count="
        f"{numeric_block_union_count}"
    )
    print(
        "global_visual_edges="
        f"{global_visual_edges}"
    )
    print(
        "boundary_visual_edges="
        f"{boundary_visual_edges}"
    )

    for split in SPLITS:
        info = split_summary[split]

        print(
            f"{split}_summary="
            f"images:{info['images']},"
            f"labels:{info['labels']},"
            f"ratio:{info['ratio']:.6f},"
            f"negative:{info['negative_images']},"
            f"person:"
            f"{info['class_counts']['person']},"
            f"helmet:"
            f"{info['class_counts']['helmet']},"
            f"vest:"
            f"{info['class_counts']['safety_vest']}"
        )

    print(
        "cross_split_dhash_pairs="
        f"{len(cross_split_close_pairs)}"
    )
    print(
        "dataset_yaml="
        f"{dataset_out / 'construction_ppe3.yaml'}"
    )
    print(
        "group_manifest="
        f"{report_out / 'group_split_manifest.csv'}"
    )
    print(
        "summary_json="
        f"{report_out / 'summary.json'}"
    )
    print(
        "exp02_4b_command_completed=YES"
    )


if __name__ == "__main__":
    main()
