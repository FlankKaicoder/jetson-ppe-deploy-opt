#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


SPLITS = ("train", "val", "test")

TARGET_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "safety_vest",
}

RANDOM_SEED = 42
GREEDY_RESTARTS = 800
LOCAL_SEARCH_STEPS = 30000
AUDIT_DHASH_THRESHOLD = 2


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
        first_root = self.find(first)
        second_root = self.find(second)

        if first_root == second_root:
            return

        if (
            self.rank[first_root]
            < self.rank[second_root]
        ):
            first_root, second_root = (
                second_root,
                first_root,
            )

        self.parent[second_root] = first_root

        if (
            self.rank[first_root]
            == self.rank[second_root]
        ):
            self.rank[first_root] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--v2-manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--reviewed-pairs",
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
        shutil.copy2(source, destination)
        return "copy"


def build_statistics(
    groups: list[dict[str, Any]],
    assignment: list[str],
) -> dict[str, dict[str, Any]]:
    statistics = {
        split: {
            "images": 0,
            "negative_images": 0,
            "classes": Counter(),
            "group_count": 0,
        }
        for split in SPLITS
    }

    for group, split in zip(
        groups,
        assignment,
    ):
        statistics[split]["images"] += int(
            group["image_count"]
        )

        statistics[split][
            "negative_images"
        ] += int(
            group["negative_count"]
        )

        statistics[split]["classes"].update(
            group["class_counts"]
        )

        statistics[split]["group_count"] += 1

    return statistics


def calculate_score(
    groups: list[dict[str, Any]],
    assignment: list[str],
    total_images: int,
    total_classes: Counter[int],
    total_negatives: int,
) -> float:
    statistics = build_statistics(
        groups,
        assignment,
    )

    score = 0.0

    for split in SPLITS:
        ratio = TARGET_RATIOS[split]

        target_images = total_images * ratio

        image_error = (
            statistics[split]["images"]
            - target_images
        ) / max(target_images, 1.0)

        # 图像数量是第一优先级。
        score += 8.0 * image_error * image_error

        for class_id in CLASS_NAMES:
            target_class = (
                total_classes[class_id]
                * ratio
            )

            class_error = (
                statistics[split]["classes"][
                    class_id
                ]
                - target_class
            ) / max(target_class, 1.0)

            score += (
                1.5
                * class_error
                * class_error
            )

        target_negative = (
            total_negatives * ratio
        )

        negative_error = (
            statistics[split][
                "negative_images"
            ]
            - target_negative
        ) / max(target_negative, 1.0)

        score += (
            0.10
            * negative_error
            * negative_error
        )

        actual_ratio = (
            statistics[split]["images"]
            / total_images
        )

        # 超出合理窗口时加入额外惩罚。
        if split == "train":
            lower, upper = 0.68, 0.72
        else:
            lower, upper = 0.13, 0.17

        if actual_ratio < lower:
            score += (
                500.0
                * (lower - actual_ratio) ** 2
            )

        if actual_ratio > upper:
            score += (
                500.0
                * (actual_ratio - upper) ** 2
            )

    return score


def create_initial_assignment(
    groups: list[dict[str, Any]],
    rng: random.Random,
    total_images: int,
    total_classes: Counter[int],
    total_negatives: int,
) -> list[str]:
    order = list(range(len(groups)))

    rng.shuffle(order)

    order.sort(
        key=lambda index: (
            -int(groups[index]["image_count"]),
            -sum(
                groups[index][
                    "class_counts"
                ].values()
            ),
        )
    )

    assignment: list[str | None] = [
        None
        for _ in groups
    ]

    # 确保三个划分都至少拥有一个组。
    seeded_splits = list(SPLITS)
    rng.shuffle(seeded_splits)

    for position, split in zip(
        order[:3],
        seeded_splits,
    ):
        assignment[position] = split

    for group_index in order[3:]:
        candidate_results = []

        for split in SPLITS:
            trial = [
                value
                if value is not None
                else ""
                for value in assignment
            ]

            trial[group_index] = split

            # 未分配的组暂时忽略。
            partial_groups = []
            partial_assignment = []

            for index, assigned_split in enumerate(
                trial
            ):
                if not assigned_split:
                    continue

                partial_groups.append(
                    groups[index]
                )

                partial_assignment.append(
                    assigned_split
                )

            assigned_images = sum(
                int(group["image_count"])
                for group in partial_groups
            )

            assigned_classes: Counter[int] = (
                Counter()
            )

            assigned_negatives = 0

            for group in partial_groups:
                assigned_classes.update(
                    group["class_counts"]
                )

                assigned_negatives += int(
                    group["negative_count"]
                )

            partial_score = calculate_score(
                partial_groups,
                partial_assignment,
                max(assigned_images, 1),
                assigned_classes,
                assigned_negatives,
            )

            current_images = sum(
                int(groups[index]["image_count"])
                for index, value
                in enumerate(assignment)
                if value == split
            )

            target_fill = (
                current_images
                / max(
                    total_images
                    * TARGET_RATIOS[split],
                    1.0,
                )
            )

            candidate_results.append(
                (
                    partial_score,
                    target_fill,
                    rng.random(),
                    split,
                )
            )

        assignment[group_index] = min(
            candidate_results
        )[-1]

    return [
        str(value)
        for value in assignment
    ]


def optimize_assignment(
    groups: list[dict[str, Any]],
) -> tuple[list[str], float]:
    rng = random.Random(RANDOM_SEED)

    total_images = sum(
        int(group["image_count"])
        for group in groups
    )

    total_classes: Counter[int] = Counter()
    total_negatives = 0

    for group in groups:
        total_classes.update(
            group["class_counts"]
        )

        total_negatives += int(
            group["negative_count"]
        )

    best_assignment: list[str] | None = None
    best_score = math.inf

    for _ in range(GREEDY_RESTARTS):
        assignment = create_initial_assignment(
            groups,
            rng,
            total_images,
            total_classes,
            total_negatives,
        )

        score = calculate_score(
            groups,
            assignment,
            total_images,
            total_classes,
            total_negatives,
        )

        if score < best_score:
            best_score = score
            best_assignment = list(assignment)

    if best_assignment is None:
        raise RuntimeError(
            "Failed to generate initial assignment"
        )

    current = list(best_assignment)
    current_score = best_score

    for step in range(LOCAL_SEARCH_STEPS):
        trial = list(current)

        if rng.random() < 0.60:
            group_index = rng.randrange(
                len(groups)
            )

            old_split = trial[group_index]

            choices = [
                split
                for split in SPLITS
                if split != old_split
            ]

            new_split = rng.choice(choices)

            # 不允许某个划分失去最后一个组。
            if (
                trial.count(old_split)
                <= 1
            ):
                continue

            trial[group_index] = new_split

        else:
            first = rng.randrange(
                len(groups)
            )

            second = rng.randrange(
                len(groups)
            )

            if (
                first == second
                or trial[first] == trial[second]
            ):
                continue

            trial[first], trial[second] = (
                trial[second],
                trial[first],
            )

        trial_score = calculate_score(
            groups,
            trial,
            total_images,
            total_classes,
            total_negatives,
        )

        temperature = max(
            0.0001,
            0.03
            * (
                1.0
                - step / LOCAL_SEARCH_STEPS
            ),
        )

        delta = trial_score - current_score

        if (
            delta <= 0.0
            or rng.random()
            < math.exp(
                -delta / temperature
            )
        ):
            current = trial
            current_score = trial_score

            if current_score < best_score:
                best_score = current_score
                best_assignment = list(current)

    return best_assignment, best_score


def main() -> None:
    args = parse_args()

    v2_manifest = args.v2_manifest.resolve()
    reviewed_pairs_path = (
        args.reviewed_pairs.resolve()
    )

    dataset_out = args.dataset_out.resolve()
    report_out = args.report_out.resolve()

    if not v2_manifest.is_file():
        raise SystemExit(
            f"ERROR: manifest not found: "
            f"{v2_manifest}"
        )

    if not reviewed_pairs_path.is_file():
        raise SystemExit(
            f"ERROR: reviewed-pair CSV not found: "
            f"{reviewed_pairs_path}"
        )

    if dataset_out.exists():
        raise SystemExit(
            f"ERROR: output already exists: "
            f"{dataset_out}"
        )

    report_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = read_csv(v2_manifest)
    reviewed_pairs = read_csv(
        reviewed_pairs_path
    )

    if len(records) != 1416:
        raise RuntimeError(
            f"Expected 1416 records, "
            f"found {len(records)}"
        )

    # v2 的 group_id 只表示数字连续块，
    # 与 v2 最终分到了哪个集合无关。
    base_group_ids = sorted(
        {
            row["group_id"]
            for row in records
        }
    )

    group_index = {
        group_id: index
        for index, group_id
        in enumerate(base_group_ids)
    }

    union_find = UnionFind(
        len(base_group_ids)
    )

    source_to_group: dict[str, str] = {}

    for row in records:
        source_path = str(
            Path(row["source_image"]).resolve()
        )

        source_to_group[
            source_path
        ] = row["group_id"]

    reviewed_edges_used = 0
    missing_reviewed_paths: list[str] = []

    for pair in reviewed_pairs:
        first_path = str(
            Path(pair["image_path_a"]).resolve()
        )

        second_path = str(
            Path(pair["image_path_b"]).resolve()
        )

        first_group = source_to_group.get(
            first_path
        )

        second_group = source_to_group.get(
            second_path
        )

        if (
            first_group is None
            or second_group is None
        ):
            missing_reviewed_paths.append(
                f"{first_path} | {second_path}"
            )

            continue

        union_find.union(
            group_index[first_group],
            group_index[second_group],
        )

        reviewed_edges_used += 1

    if missing_reviewed_paths:
        raise RuntimeError(
            "Some reviewed image paths were not "
            "found in the canonical manifest:\n"
            + "\n".join(missing_reviewed_paths)
        )

    merged_group_records: dict[
        int,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in records:
        root = union_find.find(
            group_index[row["group_id"]]
        )

        merged_group_records[root].append(
            row
        )

    groups: list[dict[str, Any]] = []

    for group_number, group_rows in enumerate(
        sorted(
            merged_group_records.values(),
            key=lambda rows: (
                -len(rows),
                rows[0]["group_id"],
            ),
        ),
        start=1,
    ):
        class_counts: Counter[int] = Counter()
        negative_count = 0

        for row in group_rows:
            person = int(row["person_boxes"])
            helmet = int(row["helmet_boxes"])
            vest = int(
                row["safety_vest_boxes"]
            )

            class_counts[0] += person
            class_counts[1] += helmet
            class_counts[2] += vest

            if (
                person + helmet + vest
                == 0
            ):
                negative_count += 1

        groups.append(
            {
                "group_id": (
                    f"reviewed_group_"
                    f"{group_number:04d}"
                ),
                "records": group_rows,
                "image_count": len(group_rows),
                "class_counts": class_counts,
                "negative_count": negative_count,
            }
        )

    assignment, optimization_score = (
        optimize_assignment(groups)
    )

    statistics = build_statistics(
        groups,
        assignment,
    )

    total_images = sum(
        int(group["image_count"])
        for group in groups
    )

    total_classes: Counter[int] = Counter()

    for group in groups:
        total_classes.update(
            group["class_counts"]
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

    manifest_rows: list[dict[str, Any]] = []
    record_assignment: dict[str, str] = {}
    transfer_counts: Counter[str] = Counter()

    for group, split in zip(
        groups,
        assignment,
    ):
        for row in group["records"]:
            image_source = Path(
                row["canonical_image"]
            ).resolve()

            label_source = Path(
                row["canonical_label"]
            ).resolve()

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

            source_path = str(
                Path(
                    row["source_image"]
                ).resolve()
            )

            record_assignment[
                source_path
            ] = split

            manifest_rows.append(
                {
                    "reviewed_group_id": (
                        group["group_id"]
                    ),
                    "assigned_split": split,
                    "group_size": (
                        group["image_count"]
                    ),
                    "source_image": (
                        source_path
                    ),
                    "canonical_image": str(
                        image_source
                    ),
                    "canonical_label": str(
                        label_source
                    ),
                    "person_boxes": (
                        row["person_boxes"]
                    ),
                    "helmet_boxes": (
                        row["helmet_boxes"]
                    ),
                    "safety_vest_boxes": (
                        row[
                            "safety_vest_boxes"
                        ]
                    ),
                    "is_negative": (
                        row["is_negative"]
                    ),
                    "dhash": row["dhash"],
                    "image_transfer": (
                        image_action
                    ),
                    "label_transfer": (
                        label_action
                    ),
                }
            )

    reviewed_cross_split = []

    for pair in reviewed_pairs:
        first_path = str(
            Path(pair["image_path_a"]).resolve()
        )

        second_path = str(
            Path(pair["image_path_b"]).resolve()
        )

        first_split = record_assignment[
            first_path
        ]

        second_split = record_assignment[
            second_path
        ]

        if first_split != second_split:
            reviewed_cross_split.append(
                {
                    **pair,
                    "assigned_split_a": (
                        first_split
                    ),
                    "assigned_split_b": (
                        second_split
                    ),
                }
            )

    dhash_rows = [
        {
            "source_image": row[
                "source_image"
            ],
            "split": row[
                "assigned_split"
            ],
            "dhash": int(
                row["dhash"],
                16,
            ),
        }
        for row in manifest_rows
    ]

    cross_split_dhash_pairs = []

    for first_index in range(
        len(dhash_rows)
    ):
        first = dhash_rows[first_index]

        for second_index in range(
            first_index + 1,
            len(dhash_rows),
        ):
            second = dhash_rows[second_index]

            if first["split"] == second["split"]:
                continue

            distance = (
                int(first["dhash"])
                ^ int(second["dhash"])
            ).bit_count()

            if distance <= AUDIT_DHASH_THRESHOLD:
                cross_split_dhash_pairs.append(
                    {
                        "hamming_distance": distance,
                        "split_a": first["split"],
                        "image_a": (
                            first["source_image"]
                        ),
                        "split_b": second["split"],
                        "image_b": (
                            second["source_image"]
                        ),
                    }
                )

    split_summary: dict[str, Any] = {}

    validation_errors = []

    for split in SPLITS:
        info = statistics[split]

        image_ratio = (
            info["images"]
            / total_images
        )

        class_ratios = {
            class_id: (
                info["classes"][class_id]
                / total_classes[class_id]
            )
            for class_id in CLASS_NAMES
        }

        split_summary[split] = {
            "images": info["images"],
            "labels": info["images"],
            "image_ratio": image_ratio,
            "group_count": (
                info["group_count"]
            ),
            "negative_images": (
                info["negative_images"]
            ),
            "class_counts": {
                CLASS_NAMES[class_id]: (
                    info["classes"][
                        class_id
                    ]
                )
                for class_id in CLASS_NAMES
            },
            "class_ratios": {
                CLASS_NAMES[class_id]: (
                    class_ratios[class_id]
                )
                for class_id in CLASS_NAMES
            },
        }

        target = TARGET_RATIOS[split]

        if abs(image_ratio - target) > 0.03:
            validation_errors.append(
                f"{split} image ratio "
                f"{image_ratio:.6f} differs "
                f"from target {target:.6f}"
            )

        for class_id in CLASS_NAMES:
            if (
                abs(
                    class_ratios[class_id]
                    - target
                )
                > 0.08
            ):
                validation_errors.append(
                    f"{split} "
                    f"{CLASS_NAMES[class_id]} "
                    f"ratio "
                    f"{class_ratios[class_id]:.6f} "
                    f"differs from target "
                    f"{target:.6f}"
                )

    if reviewed_cross_split:
        validation_errors.append(
            "Reviewed near-duplicate edges "
            "still cross splits: "
            f"{len(reviewed_cross_split)}"
        )

    if cross_split_dhash_pairs:
        validation_errors.append(
            "Cross-split dHash<=2 pairs remain: "
            f"{len(cross_split_dhash_pairs)}"
        )

    dataset_yaml = {
        "path": str(dataset_out),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
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
        / "reviewed_split_manifest.csv",
        manifest_rows,
        [
            "reviewed_group_id",
            "assigned_split",
            "group_size",
            "source_image",
            "canonical_image",
            "canonical_label",
            "person_boxes",
            "helmet_boxes",
            "safety_vest_boxes",
            "is_negative",
            "dhash",
            "image_transfer",
            "label_transfer",
        ],
    )

    write_csv(
        report_out
        / "reviewed_edges_still_cross_split.csv",
        reviewed_cross_split,
        list(reviewed_cross_split[0].keys())
        if reviewed_cross_split
        else [
            "hamming_distance",
            "split_a",
            "image_key_a",
            "image_path_a",
            "split_b",
            "image_key_b",
            "image_path_b",
            "assigned_split_a",
            "assigned_split_b",
        ],
    )

    write_csv(
        report_out
        / "cross_split_dhash_pairs.csv",
        cross_split_dhash_pairs,
        [
            "hamming_distance",
            "split_a",
            "image_a",
            "split_b",
            "image_b",
        ],
    )

    metadata = {
        "split_execution": (
            "PASS"
            if not validation_errors
            else "REVIEW_REQUIRED"
        ),
        "v2_manifest": str(v2_manifest),
        "reviewed_pairs": str(
            reviewed_pairs_path
        ),
        "dataset_out": str(dataset_out),
        "random_seed": RANDOM_SEED,
        "greedy_restarts": GREEDY_RESTARTS,
        "local_search_steps": (
            LOCAL_SEARCH_STEPS
        ),
        "optimization_score": (
            optimization_score
        ),
        "base_group_count": len(
            base_group_ids
        ),
        "reviewed_edges_used": (
            reviewed_edges_used
        ),
        "merged_group_count": len(groups),
        "max_group_size": max(
            int(group["image_count"])
            for group in groups
        ),
        "total_images": total_images,
        "global_class_counts": {
            CLASS_NAMES[class_id]: (
                total_classes[class_id]
            )
            for class_id in CLASS_NAMES
        },
        "split_summary": split_summary,
        "reviewed_cross_split_edges": (
            len(reviewed_cross_split)
        ),
        "cross_split_dhash_pairs": (
            len(cross_split_dhash_pairs)
        ),
        "validation_errors": (
            validation_errors
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

    if not validation_errors:
        (
            dataset_out
            / ".exp02_4c_complete"
        ).write_text(
            "PASS\n",
            encoding="utf-8",
        )

    print(
        "============================================================"
    )
    print(
        " Exp02.4c Reviewed Balanced Split Summary"
    )
    print(
        "============================================================"
    )
    print(
        "split_execution="
        + (
            "PASS"
            if not validation_errors
            else "REVIEW_REQUIRED"
        )
    )
    print(
        f"dataset_out={dataset_out}"
    )
    print(
        f"total_images={total_images}"
    )
    print(
        f"base_group_count="
        f"{len(base_group_ids)}"
    )
    print(
        f"reviewed_edges_used="
        f"{reviewed_edges_used}"
    )
    print(
        f"merged_group_count="
        f"{len(groups)}"
    )
    print(
        "max_group_size="
        f"{max(group['image_count'] for group in groups)}"
    )
    print(
        f"optimization_score="
        f"{optimization_score:.10f}"
    )

    for split in SPLITS:
        info = split_summary[split]

        print(
            f"{split}_summary="
            f"images:{info['images']},"
            f"ratio:{info['image_ratio']:.6f},"
            f"groups:{info['group_count']},"
            f"negative:{info['negative_images']},"
            f"person:"
            f"{info['class_counts']['person']},"
            f"person_ratio:"
            f"{info['class_ratios']['person']:.6f},"
            f"helmet:"
            f"{info['class_counts']['helmet']},"
            f"helmet_ratio:"
            f"{info['class_ratios']['helmet']:.6f},"
            f"vest:"
            f"{info['class_counts']['safety_vest']},"
            f"vest_ratio:"
            f"{info['class_ratios']['safety_vest']:.6f}"
        )

    print(
        "reviewed_cross_split_edges="
        f"{len(reviewed_cross_split)}"
    )
    print(
        "cross_split_dhash_pairs="
        f"{len(cross_split_dhash_pairs)}"
    )
    print(
        "validation_error_count="
        f"{len(validation_errors)}"
    )

    for index, error in enumerate(
        validation_errors,
        start=1,
    ):
        print(
            f"validation_error_{index}="
            f"{error}"
        )

    print(
        "dataset_yaml="
        f"{dataset_out / 'construction_ppe3.yaml'}"
    )
    print(
        "split_manifest="
        f"{report_out / 'reviewed_split_manifest.csv'}"
    )
    print(
        "summary_json="
        f"{report_out / 'summary.json'}"
    )
    print(
        "exp02_4c_command_completed=YES"
    )


if __name__ == "__main__":
    main()
