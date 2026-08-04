#!/usr/bin/env bash

set -Eeuo pipefail

REPO_DIR="/root/autodl-tmp/jetson-ppe-deploy-opt"

DATA_BASE="/root/autodl-tmp/datasets"
DOWNLOAD_DIR="${DATA_BASE}/downloads"
SOURCE_DIR="${DATA_BASE}/sources"

DATASET_ID="construction-ppe"
DATASET_VERSION="ultralytics_2025_v1"

ARCHIVE="${DOWNLOAD_DIR}/${DATASET_ID}.zip"
TARGET_DIR="${SOURCE_DIR}/${DATASET_ID}_${DATASET_VERSION}"

SOURCE_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"

RUNTIME_DIR="/root/autodl-tmp/jetson-ppe-runtime"
export YOLO_CONFIG_DIR="${RUNTIME_DIR}/ultralytics"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO_DIR}/results/dataset_audit/exp02_2_dataset_acquisition_${TIMESTAMP}"

LOG="${OUT_DIR}/run.log"
SUMMARY="${OUT_DIR}/summary.txt"
ABNORMAL="${OUT_DIR}/abnormal.txt"

mkdir -p \
    "$DOWNLOAD_DIR" \
    "$SOURCE_DIR" \
    "$YOLO_CONFIG_DIR" \
    "$OUT_DIR"

exec > >(tee "$LOG") 2>&1

cleanup()
{
    if [ -n "${TEMP_DIR:-}" ] && [ -d "${TEMP_DIR:-}" ]; then
        rm -rf "$TEMP_DIR"
    fi
}

on_error()
{
    local rc=$?
    local line="${BASH_LINENO[0]:-UNKNOWN}"
    local command="${BASH_COMMAND:-UNKNOWN}"

    echo
    echo "============================================================"
    echo " Exp02.2 failed"
    echo "============================================================"
    echo "return_code=$rc"
    echo "failed_line=$line"
    echo "failed_command=$command"
    echo "log=$LOG"

    exit "$rc"
}

trap cleanup EXIT
trap on_error ERR

cd "$REPO_DIR"

echo "============================================================"
echo " Exp02.2: Construction-PPE Dataset Acquisition"
echo "============================================================"
echo "timestamp=$(date --iso-8601=seconds)"
echo "repo_dir=$REPO_DIR"
echo "data_base=$DATA_BASE"
echo "archive=$ARCHIVE"
echo "target_dir=$TARGET_DIR"
echo "source_url=$SOURCE_URL"
echo "out_dir=$OUT_DIR"

echo
echo "========== environment =========="

echo "python=$(command -v python)"
echo "python_version=$(python --version 2>&1)"
echo "yolo_config_dir=$YOLO_CONFIG_DIR"
echo "git_branch=$(git branch --show-current)"
echo "git_commit=$(git rev-parse HEAD)"

echo
echo "========== storage before download =========="

df -h /root/autodl-tmp

AVAILABLE_KB="$(
    df --output=avail /root/autodl-tmp \
        | tail -1 \
        | tr -d ' '
)"

if [ -z "$AVAILABLE_KB" ]; then
    echo "ERROR: cannot determine free disk space"
    exit 10
fi

AVAILABLE_MB=$((AVAILABLE_KB / 1024))

echo "available_mb=$AVAILABLE_MB"

if [ "$AVAILABLE_MB" -lt 2048 ]; then
    echo "ERROR: less than 2 GB free space"
    exit 11
fi

echo
echo "========== download archive =========="

if [ -f "$ARCHIVE" ]; then
    echo "archive_action=REUSE_EXISTING"
    ls -lh "$ARCHIVE"
else
    echo "archive_action=DOWNLOAD"

    PART_FILE="${ARCHIVE}.part_${TIMESTAMP}"

    if command -v curl >/dev/null 2>&1; then
        curl \
            --fail \
            --location \
            --retry 5 \
            --retry-delay 3 \
            --connect-timeout 30 \
            --output "$PART_FILE" \
            "$SOURCE_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget \
            --tries=5 \
            --timeout=30 \
            --output-document="$PART_FILE" \
            "$SOURCE_URL"
    else
        echo "ERROR: neither curl nor wget is available"
        exit 20
    fi

    if [ ! -s "$PART_FILE" ]; then
        echo "ERROR: downloaded archive is empty"
        exit 21
    fi

    mv "$PART_FILE" "$ARCHIVE"

    ls -lh "$ARCHIVE"
fi

echo
echo "========== archive checksum =========="

ARCHIVE_SHA256="$(
    sha256sum "$ARCHIVE" \
        | awk '{print $1}'
)"

ARCHIVE_BYTES="$(
    stat --format='%s' "$ARCHIVE"
)"

echo "archive_bytes=$ARCHIVE_BYTES"
echo "archive_sha256=$ARCHIVE_SHA256"

printf '%s  %s\n' \
    "$ARCHIVE_SHA256" \
    "$ARCHIVE" \
    > "$OUT_DIR/archive_sha256.txt"

echo
echo "========== verify ZIP integrity =========="

python - "$ARCHIVE" "$OUT_DIR/archive_listing.txt" <<'PY'
from pathlib import Path
from zipfile import BadZipFile, ZipFile
import sys

archive = Path(sys.argv[1])
listing_path = Path(sys.argv[2])

try:
    with ZipFile(archive, "r") as zip_file:
        bad_member = zip_file.testzip()

        if bad_member is not None:
            raise SystemExit(
                f"ERROR: corrupted ZIP member: {bad_member}"
            )

        members = zip_file.infolist()

        with listing_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for member in members:
                file.write(
                    f"{member.file_size}\t{member.filename}\n"
                )

        print(f"archive_member_count={len(members)}")
        print("archive_integrity=PASS")

except BadZipFile as exc:
    raise SystemExit(f"ERROR: invalid ZIP archive: {exc}")
PY

echo
echo "========== extract dataset =========="

if [ -f "$TARGET_DIR/.exp02_2_extract_complete" ]; then
    echo "extract_action=REUSE_COMPLETE_DATASET"
else
    if [ -e "$TARGET_DIR" ]; then
        echo "ERROR: target exists but has no completion marker"
        echo "target_dir=$TARGET_DIR"
        echo "No files were overwritten."
        exit 30
    fi

    TEMP_DIR="${TARGET_DIR}.tmp_${TIMESTAMP}"

    mkdir -p "$TEMP_DIR"

    python - "$ARCHIVE" "$TEMP_DIR" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import sys

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])

with ZipFile(archive, "r") as zip_file:
    zip_file.extractall(destination)

print(f"temporary_extract_dir={destination}")
print("archive_extract=PASS")
PY

    mapfile -t DATASET_CANDIDATES < <(
        find "$TEMP_DIR" \
            -type d \
            -path '*/images/train' \
            -print \
        | while read -r train_dir
          do
              dirname "$(dirname "$train_dir")"
          done \
        | sort -u
    )

    echo "dataset_candidate_count=${#DATASET_CANDIDATES[@]}"

    if [ "${#DATASET_CANDIDATES[@]}" -ne 1 ]; then
        echo "ERROR: expected exactly one dataset root"
        echo
        echo "Detected candidates:"

        printf '%s\n' "${DATASET_CANDIDATES[@]:-NONE}"

        echo
        echo "Extracted directory preview:"

        find "$TEMP_DIR" \
            -maxdepth 4 \
            -printf '%y %p\n' \
            | sort \
            | head -200

        exit 31
    fi

    DETECTED_ROOT="${DATASET_CANDIDATES[0]}"

    echo "detected_dataset_root=$DETECTED_ROOT"

    mv "$DETECTED_ROOT" "$TARGET_DIR"

    touch "$TARGET_DIR/.exp02_2_extract_complete"
fi

echo
echo "========== dataset structure =========="

find "$TARGET_DIR" \
    -maxdepth 3 \
    -type d \
    | sort \
    | tee "$OUT_DIR/directory_tree.txt"

echo
echo "========== raw dataset audit =========="

python - \
    "$TARGET_DIR" \
    "$OUT_DIR/raw_dataset_inventory.json" \
    "$OUT_DIR/yaml_inventory.txt" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

dataset_root = Path(sys.argv[1]).resolve()
json_output = Path(sys.argv[2])
yaml_output = Path(sys.argv[3])

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

inventory: dict[str, object] = {
    "dataset_root": str(dataset_root),
    "splits": {},
    "yaml_files": [],
}

for split in ("train", "val", "test"):
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    images = []

    if image_dir.is_dir():
        images = sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in image_extensions
        )

    labels = []

    if label_dir.is_dir():
        labels = sorted(label_dir.rglob("*.txt"))

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    split_info = {
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "image_count": len(images),
        "label_count": len(labels),
        "images_without_labels": sorted(
            image_stems - label_stems
        ),
        "labels_without_images": sorted(
            label_stems - image_stems
        ),
    }

    inventory["splits"][split] = split_info

    print(f"{split}_image_count={len(images)}")
    print(f"{split}_label_count={len(labels)}")
    print(
        f"{split}_images_without_labels="
        f"{len(split_info['images_without_labels'])}"
    )
    print(
        f"{split}_labels_without_images="
        f"{len(split_info['labels_without_images'])}"
    )

yaml_files = sorted(
    list(dataset_root.rglob("*.yaml"))
    + list(dataset_root.rglob("*.yml"))
)

with yaml_output.open("w", encoding="utf-8") as file:
    if not yaml_files:
        file.write("No YAML files found.\n")

    for yaml_path in yaml_files:
        record: dict[str, object] = {
            "path": str(yaml_path),
        }

        file.write("=" * 80 + "\n")
        file.write(f"path={yaml_path}\n")

        try:
            content = yaml.safe_load(
                yaml_path.read_text(encoding="utf-8")
            )

            record["content"] = content

            file.write(
                yaml.safe_dump(
                    content,
                    allow_unicode=True,
                    sort_keys=False,
                )
            )

            if isinstance(content, dict):
                names = content.get("names")

                if isinstance(names, dict):
                    print(
                        "class_names="
                        + ",".join(
                            f"{key}:{value}"
                            for key, value in names.items()
                        )
                    )
                elif isinstance(names, list):
                    print(
                        "class_names="
                        + ",".join(
                            f"{index}:{value}"
                            for index, value in enumerate(names)
                        )
                    )

        except Exception as exc:
            record["parse_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            file.write(
                f"parse_error={record['parse_error']}\n"
            )

        inventory["yaml_files"].append(record)

json_output.write_text(
    json.dumps(
        inventory,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

total_images = sum(
    int(split["image_count"])
    for split in inventory["splits"].values()
)

total_labels = sum(
    int(split["label_count"])
    for split in inventory["splits"].values()
)

print(f"total_image_count={total_images}")
print(f"total_label_count={total_labels}")
print(f"yaml_file_count={len(yaml_files)}")

if total_images == 0:
    raise SystemExit("ERROR: no images detected")

if total_labels == 0:
    raise SystemExit("ERROR: no labels detected")

print("raw_dataset_inventory=PASS")
PY

echo
echo "========== save source record =========="

cat > "$OUT_DIR/dataset_source.txt" <<EOF
dataset_id=$DATASET_ID
dataset_version=$DATASET_VERSION
source_url=$SOURCE_URL
archive=$ARCHIVE
archive_bytes=$ARCHIVE_BYTES
archive_sha256=$ARCHIVE_SHA256
dataset_root=$TARGET_DIR
download_date=$(date --iso-8601=seconds)
declared_license=AGPL-3.0
declared_image_count=1416
declared_train_count=1132
declared_val_count=143
declared_test_count=141
declared_classes=helmet,gloves,vest,boots,goggles,none,Person,no_helmet,no_goggle,no_gloves,no_boots
EOF

TRAIN_IMAGES="$(
    find "$TARGET_DIR/images/train" \
        -type f \
        \( \
            -iname '*.jpg' \
            -o -iname '*.jpeg' \
            -o -iname '*.png' \
            -o -iname '*.bmp' \
            -o -iname '*.webp' \
        \) \
        | wc -l
)"

VAL_IMAGES="$(
    find "$TARGET_DIR/images/val" \
        -type f \
        \( \
            -iname '*.jpg' \
            -o -iname '*.jpeg' \
            -o -iname '*.png' \
            -o -iname '*.bmp' \
            -o -iname '*.webp' \
        \) \
        | wc -l
)"

TEST_IMAGES="$(
    find "$TARGET_DIR/images/test" \
        -type f \
        \( \
            -iname '*.jpg' \
            -o -iname '*.jpeg' \
            -o -iname '*.png' \
            -o -iname '*.bmp' \
            -o -iname '*.webp' \
        \) \
        | wc -l
)"

TOTAL_IMAGES=$((TRAIN_IMAGES + VAL_IMAGES + TEST_IMAGES))

echo
echo "========== generate summary =========="

{
    echo "============================================================"
    echo " Exp02.2 Summary"
    echo "============================================================"
    echo "result=PASS"
    echo "dataset_id=$DATASET_ID"
    echo "dataset_version=$DATASET_VERSION"
    echo "dataset_root=$TARGET_DIR"
    echo "archive=$ARCHIVE"
    echo "archive_bytes=$ARCHIVE_BYTES"
    echo "archive_sha256=$ARCHIVE_SHA256"
    echo "train_images=$TRAIN_IMAGES"
    echo "val_images=$VAL_IMAGES"
    echo "test_images=$TEST_IMAGES"
    echo "total_images=$TOTAL_IMAGES"
    echo "source_record=$OUT_DIR/dataset_source.txt"
    echo "inventory_json=$OUT_DIR/raw_dataset_inventory.json"
    echo "yaml_inventory=$OUT_DIR/yaml_inventory.txt"
    echo "run_log=$LOG"
} | tee "$SUMMARY"

grep -nEi \
    'traceback|badzipfile|corrupted ZIP|segmentation fault|no images detected|no labels detected|archive_integrity=FAIL|raw_dataset_inventory=FAIL|ERROR:' \
    "$LOG" \
    > "$ABNORMAL" || true

if [ ! -s "$ABNORMAL" ]; then
    echo "No abnormal messages detected." > "$ABNORMAL"
fi

echo
echo "========== summary =========="
cat "$SUMMARY"

echo
echo "========== abnormal =========="
cat "$ABNORMAL"

echo
echo "exp02_2_command_completed=YES"
echo "exp02_2_out_dir=$OUT_DIR"
