#!/usr/bin/env bash

set -uo pipefail

REPO_DIR="/root/autodl-tmp/jetson-ppe-deploy-opt"
VENV_DIR="$REPO_DIR/.venv-autodl"

RUNTIME_DIR="/root/autodl-tmp/jetson-ppe-runtime"
MODEL_DIR="/root/autodl-tmp/models/ultralytics"
OUTPUT_ROOT="/root/autodl-tmp/jetson-ppe-outputs"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

REPORT_DIR="$REPO_DIR/results/training/exp02_5_yolo11n_smoke_${TIMESTAMP}"
TRAIN_DIR="$OUTPUT_ROOT/exp02_5_yolo11n_smoke_${TIMESTAMP}"

LOG="$REPORT_DIR/run.log"
ABNORMAL="$REPORT_DIR/abnormal.txt"

mkdir -p \
    "$REPORT_DIR" \
    "$MODEL_DIR" \
    "$OUTPUT_ROOT" \
    "$RUNTIME_DIR/ultralytics"

cd "$REPO_DIR" || {
    echo "ERROR: repository not found: $REPO_DIR"
    exit 2
}

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "ERROR: virtual environment not found"
    exit 3
fi

source "$VENV_DIR/bin/activate"

export YOLO_CONFIG_DIR="$RUNTIME_DIR/ultralytics"
export PYTHONUNBUFFERED=1

# 只影响当前脚本，用于下载预训练权重。
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo
fi

FINAL_REPORT="$(
    find \
        "$REPO_DIR/results/dataset_audit" \
        -maxdepth 1 \
        -type d \
        -name 'exp02_4e_final_split_*' \
        -printf '%T@ %p\n' \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
)"

if [ -z "$FINAL_REPORT" ]; then
    echo "ERROR: Exp02.4e report not found"
    exit 4
fi

SUMMARY_JSON="$FINAL_REPORT/summary.json"

if [ ! -f "$SUMMARY_JSON" ]; then
    echo "ERROR: final split summary not found"
    exit 5
fi

DATASET_ROOT="$(
    python - "$SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(
    Path(sys.argv[1]).read_text(encoding="utf-8")
)

print(summary["dataset_out"])
PY
)"

DATASET_YAML="$DATASET_ROOT/construction_ppe3.yaml"
MODEL_PATH="$MODEL_DIR/yolo11n.pt"

if [ ! -f "$DATASET_YAML" ]; then
    echo "ERROR: dataset YAML not found: $DATASET_YAML"
    exit 6
fi

echo "============================================================"
echo " Exp02.5: YOLO11n One-Epoch Training Smoke"
echo "============================================================"
echo "timestamp=$(date --iso-8601=seconds)"
echo "repo_dir=$REPO_DIR"
echo "dataset_root=$DATASET_ROOT"
echo "dataset_yaml=$DATASET_YAML"
echo "model_path=$MODEL_PATH"
echo "train_dir=$TRAIN_DIR"
echo "report_dir=$REPORT_DIR"
echo "python=$(command -v python)"
echo "git_branch=$(git branch --show-current)"
echo "git_commit=$(git rev-parse HEAD)"

python - "$DATASET_YAML" "$MODEL_PATH" "$TRAIN_DIR" "$REPORT_DIR" \
    2>&1 | tee "$LOG" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.downloads import attempt_download_asset


dataset_yaml = Path(sys.argv[1]).resolve()
model_path = Path(sys.argv[2]).resolve()
train_dir = Path(sys.argv[3]).resolve()
report_dir = Path(sys.argv[4]).resolve()

report_dir.mkdir(parents=True, exist_ok=True)
model_path.parent.mkdir(parents=True, exist_ok=True)

print("========== environment ==========")
print(f"torch_version={torch.__version__}")
print(f"torch_cuda_version={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print(
    "cuda_device="
    f"{torch.cuda.get_device_name(0)}"
)

print(
    "cuda_capability="
    f"{torch.cuda.get_device_capability(0)}"
)

print()
print("========== dataset check ==========")

dataset_info = check_det_dataset(
    str(dataset_yaml)
)

print(
    "dataset_names="
    f"{dataset_info['names']}"
)

print(
    "dataset_train="
    f"{dataset_info['train']}"
)

print(
    "dataset_val="
    f"{dataset_info['val']}"
)

print(
    "dataset_test="
    f"{dataset_info.get('test')}"
)

print("dataset_check=PASS")

print()
print("========== model acquisition ==========")

if not model_path.is_file():
    downloaded = Path(
        attempt_download_asset(
            str(model_path)
        )
    )

    if downloaded.resolve() != model_path:
        model_path.write_bytes(
            downloaded.read_bytes()
        )

if not model_path.is_file():
    raise RuntimeError(
        f"Model download failed: {model_path}"
    )

print(f"model_path={model_path}")
print(f"model_bytes={model_path.stat().st_size}")

print()
print("========== one-epoch training ==========")

model = YOLO(str(model_path))

model.train(
    data=str(dataset_yaml),
    epochs=1,
    imgsz=640,
    batch=16,
    workers=4,
    device=0,
    seed=42,
    deterministic=True,
    pretrained=True,
    optimizer="auto",
    amp=True,
    cache=False,
    plots=True,
    save=True,
    val=True,
    project=str(train_dir.parent),
    name=train_dir.name,
    exist_ok=False,
    verbose=True,
)

save_dir = Path(model.trainer.save_dir).resolve()
best_weight = save_dir / "weights" / "best.pt"
last_weight = save_dir / "weights" / "last.pt"

if not last_weight.is_file():
    raise RuntimeError(
        f"last.pt was not generated: {last_weight}"
    )

evaluation_weight = (
    best_weight
    if best_weight.is_file()
    else last_weight
)

print()
print("========== explicit validation ==========")

validation_model = YOLO(
    str(evaluation_weight)
)

metrics = validation_model.val(
    data=str(dataset_yaml),
    split="val",
    imgsz=640,
    batch=16,
    workers=4,
    device=0,
    plots=False,
    verbose=True,
)

results_dict = {
    str(key): float(value)
    for key, value
    in metrics.results_dict.items()
}

summary = {
    "experiment": (
        "Exp02.5 YOLO11n one-epoch training smoke"
    ),
    "result": "PASS",
    "dataset_yaml": str(dataset_yaml),
    "model_source": str(model_path),
    "train_save_dir": str(save_dir),
    "best_weight": (
        str(best_weight)
        if best_weight.is_file()
        else ""
    ),
    "last_weight": str(last_weight),
    "evaluation_weight": str(
        evaluation_weight
    ),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_device": torch.cuda.get_device_name(0),
    "cuda_capability": list(
        torch.cuda.get_device_capability(0)
    ),
    "epochs": 1,
    "imgsz": 640,
    "batch": 16,
    "seed": 42,
    "metrics": results_dict,
}

(report_dir / "summary.json").write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

summary_lines = [
    "============================================================",
    " Exp02.5 YOLO11n Training Smoke Summary",
    "============================================================",
    "result=PASS",
    f"dataset_yaml={dataset_yaml}",
    f"model_source={model_path}",
    f"train_save_dir={save_dir}",
    (
        "best_weight="
        + (
            str(best_weight)
            if best_weight.is_file()
            else "NOT_GENERATED"
        )
    ),
    f"last_weight={last_weight}",
    f"evaluation_weight={evaluation_weight}",
    f"torch_version={torch.__version__}",
    f"torch_cuda_version={torch.version.cuda}",
    (
        "cuda_device="
        f"{torch.cuda.get_device_name(0)}"
    ),
    "epochs=1",
    "imgsz=640",
    "batch=16",
    "seed=42",
]

for key, value in sorted(
    results_dict.items()
):
    summary_lines.append(
        f"metric_{key}={value:.8f}"
    )

summary_lines.append(
    "exp02_5_training_smoke=PASS"
)

summary_text = "\n".join(
    summary_lines
) + "\n"

(report_dir / "summary.txt").write_text(
    summary_text,
    encoding="utf-8",
)

print()
print(summary_text, end="")

PY

RC=${PIPESTATUS[0]}

grep -nEi \
    'traceback|cuda out of memory|segmentation fault|nan|inf|runtimeerror|permission denied|syntaxerror|ERROR:' \
    "$LOG" \
    > "$ABNORMAL" || true

if [ ! -s "$ABNORMAL" ]; then
    echo "No abnormal execution messages detected." \
        > "$ABNORMAL"
fi

echo
echo "========== return code =========="
echo "exp02_5_return_code=$RC"

if [ -f "$REPORT_DIR/summary.txt" ]; then
    echo
    echo "========== summary =========="
    cat "$REPORT_DIR/summary.txt"
fi

echo
echo "========== abnormal =========="
cat "$ABNORMAL"

echo
echo "train_dir=$TRAIN_DIR"
echo "report_dir=$REPORT_DIR"

if [ "$RC" -eq 0 ]; then
    echo "exp02_5_command_completed=YES"
else
    echo "exp02_5_command_completed=NO"
fi

exit "$RC"
