#!/usr/bin/env bash

set -uo pipefail

REPO_DIR="/root/autodl-tmp/jetson-ppe-deploy-opt"
VENV_DIR="$REPO_DIR/.venv-autodl"

DATA_ROOT="/root/autodl-tmp/datasets/derived/construction_ppe3_final_split_v1_20260804_175104"
DATA_YAML="$DATA_ROOT/construction_ppe3.yaml"
MODEL="/root/autodl-tmp/models/ultralytics/yolo11n.pt"
OUTPUT_ROOT="/root/autodl-tmp/jetson-ppe-outputs"

EPOCHS=100
IMGSZ=640
BATCH=16
WORKERS=8
SEED=42

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="exp02_6_yolo11n_baseline_e100_${TIMESTAMP}"

REPORT_DIR="$REPO_DIR/results/training/$RUN_NAME"
TRAIN_DIR="$OUTPUT_ROOT/$RUN_NAME"

LOG="$REPORT_DIR/run.log"
SUMMARY="$REPORT_DIR/summary.txt"
ENVIRONMENT="$REPORT_DIR/environment.txt"
ABNORMAL="$REPORT_DIR/abnormal.txt"
COMMAND_FILE="$REPORT_DIR/command.sh"

mkdir -p "$REPORT_DIR" "$OUTPUT_ROOT"

echo "============================================================"
echo " Exp02.6 YOLO11n Formal Baseline Training"
echo "============================================================"
echo "run_name=$RUN_NAME"
echo "report_dir=$REPORT_DIR"
echo "train_dir=$TRAIN_DIR"

cd "$REPO_DIR" || exit 1

if command -v flock >/dev/null 2>&1; then
    exec 9>"/tmp/jetson_ppe_exp02_6.lock"

    if ! flock -n 9; then
        echo "ERROR: another Exp02.6 process is already running"
        exit 2
    fi
fi

echo
echo "========== preflight =========="

for REQUIRED in \
    "$VENV_DIR/bin/python" \
    "$VENV_DIR/bin/yolo" \
    "$DATA_YAML" \
    "$MODEL"
do
    if [ ! -e "$REQUIRED" ]; then
        echo "ERROR: required file missing: $REQUIRED"
        exit 3
    fi

    echo "PASS: $REQUIRED"
done

if [ ! -s "$MODEL" ]; then
    echo "ERROR: model file is empty: $MODEL"
    exit 4
fi

echo
echo "========== duplicate-label preflight =========="

KNOWN_LABEL="$DATA_ROOT/labels/train/train__image187.txt"
DUPLICATE_ACTION="NOT_APPLICABLE"
DUPLICATE_LINES=0

if [ -f "$KNOWN_LABEL" ]; then
    DUPLICATE_LINES="$(
        sort "$KNOWN_LABEL" \
        | uniq -d \
        | wc -l
    )"

    if [ "$DUPLICATE_LINES" -gt 0 ]; then
        cp \
            "$KNOWN_LABEL" \
            "$REPORT_DIR/train__image187_before_fix.txt"

        awk '
        NF > 0 && !seen[$0]++ {
            print
        }
        ' "$KNOWN_LABEL" \
            > "$REPORT_DIR/train__image187_fixed.txt"

        cp \
            "$REPORT_DIR/train__image187_fixed.txt" \
            "$KNOWN_LABEL"

        rm -f "$DATA_ROOT/labels/train.cache"
        rm -f "$DATA_ROOT/labels/val.cache"

        DUPLICATE_ACTION="FIXED"
    else
        DUPLICATE_ACTION="ALREADY_CLEAN"
    fi
fi

echo "duplicate_label_lines=$DUPLICATE_LINES"
echo "duplicate_label_action=$DUPLICATE_ACTION"

echo
echo "========== environment =========="

{
    echo "time=$(date '+%F %T')"
    echo "repo_dir=$REPO_DIR"
    echo "run_name=$RUN_NAME"
    echo "data_yaml=$DATA_YAML"
    echo "model=$MODEL"
    echo "epochs=$EPOCHS"
    echo "imgsz=$IMGSZ"
    echo "batch=$BATCH"
    echo "workers=$WORKERS"
    echo "seed=$SEED"
    echo "duplicate_label_action=$DUPLICATE_ACTION"

    echo
    echo "========== git =========="
    git branch --show-current 2>/dev/null || true
    git rev-parse HEAD 2>/dev/null || true
    git status --short 2>/dev/null || true

    echo
    echo "========== python packages =========="

    "$VENV_DIR/bin/python" - <<'PY'
import sys
import torch
import ultralytics

print("python_executable =", sys.executable)
print("python_version    =", sys.version.replace("\n", " "))
print("torch_version     =", torch.__version__)
print("torch_cuda        =", torch.version.cuda)
print("ultralytics       =", ultralytics.__version__)
print("cuda_available    =", torch.cuda.is_available())

if torch.cuda.is_available():
    print("cuda_device       =", torch.cuda.get_device_name(0))
    print("cuda_capability   =", torch.cuda.get_device_capability(0))
PY

    echo
    echo "========== GPU =========="
    nvidia-smi || true

    echo
    echo "========== file hashes =========="
    sha256sum "$MODEL"
    sha256sum "$DATA_YAML"

    echo
    echo "========== dataset yaml =========="
    cat "$DATA_YAML"

} > "$ENVIRONMENT" 2>&1

cat "$ENVIRONMENT"

cat > "$COMMAND_FILE" <<EOF
#!/usr/bin/env bash

"$VENV_DIR/bin/yolo" detect train \
    model="$MODEL" \
    data="$DATA_YAML" \
    epochs=$EPOCHS \
    imgsz=$IMGSZ \
    batch=$BATCH \
    device=0 \
    workers=$WORKERS \
    seed=$SEED \
    deterministic=True \
    optimizer=AdamW \
    lr0=0.0015 \
    lrf=0.01 \
    momentum=0.9 \
    weight_decay=0.0005 \
    warmup_epochs=3.0 \
    amp=True \
    cache=False \
    patience=$EPOCHS \
    project="$OUTPUT_ROOT" \
    name="$RUN_NAME" \
    exist_ok=False \
    pretrained=True \
    val=True \
    plots=True \
    save=True \
    save_period=10 \
    verbose=True
EOF

chmod +x "$COMMAND_FILE"

echo
echo "========== frozen command =========="
cat "$COMMAND_FILE"

echo
echo "========== training start =========="

START_SECONDS="$(date +%s)"

set +e

"$VENV_DIR/bin/yolo" detect train \
    model="$MODEL" \
    data="$DATA_YAML" \
    epochs="$EPOCHS" \
    imgsz="$IMGSZ" \
    batch="$BATCH" \
    device=0 \
    workers="$WORKERS" \
    seed="$SEED" \
    deterministic=True \
    optimizer=AdamW \
    lr0=0.0015 \
    lrf=0.01 \
    momentum=0.9 \
    weight_decay=0.0005 \
    warmup_epochs=3.0 \
    amp=True \
    cache=False \
    patience="$EPOCHS" \
    project="$OUTPUT_ROOT" \
    name="$RUN_NAME" \
    exist_ok=False \
    pretrained=True \
    val=True \
    plots=True \
    save=True \
    save_period=10 \
    verbose=True \
    2>&1 | tee "$LOG"

TRAIN_RC=${PIPESTATUS[0]}

set -u

END_SECONDS="$(date +%s)"
ELAPSED_SECONDS=$((END_SECONDS - START_SECONDS))

BEST_WEIGHT="$TRAIN_DIR/weights/best.pt"
LAST_WEIGHT="$TRAIN_DIR/weights/last.pt"
RESULTS_CSV="$TRAIN_DIR/results.csv"
ARGS_YAML="$TRAIN_DIR/args.yaml"

if [ -f "$RESULTS_CSV" ]; then
    cp "$RESULTS_CSV" "$REPORT_DIR/results.csv"
fi

if [ -f "$ARGS_YAML" ]; then
    cp "$ARGS_YAML" "$REPORT_DIR/args.yaml"
fi

RESULT="FAIL"

if \
    [ "$TRAIN_RC" -eq 0 ] \
    && [ -s "$BEST_WEIGHT" ] \
    && [ -s "$LAST_WEIGHT" ] \
    && [ -s "$RESULTS_CSV" ]
then
    RESULT="PASS"
fi

{
    echo "============================================================"
    echo " Exp02.6 YOLO11n Formal Baseline Summary"
    echo "============================================================"
    echo "result=$RESULT"
    echo "train_return_code=$TRAIN_RC"
    echo "run_name=$RUN_NAME"
    echo "report_dir=$REPORT_DIR"
    echo "train_dir=$TRAIN_DIR"
    echo "data_yaml=$DATA_YAML"
    echo "model_source=$MODEL"
    echo "best_weight=$BEST_WEIGHT"
    echo "last_weight=$LAST_WEIGHT"
    echo "results_csv=$RESULTS_CSV"
    echo "epochs=$EPOCHS"
    echo "imgsz=$IMGSZ"
    echo "batch=$BATCH"
    echo "workers=$WORKERS"
    echo "seed=$SEED"
    echo "optimizer=AdamW"
    echo "lr0=0.0015"
    echo "lrf=0.01"
    echo "momentum=0.9"
    echo "weight_decay=0.0005"
    echo "elapsed_seconds=$ELAPSED_SECONDS"
    echo "duplicate_label_action=$DUPLICATE_ACTION"

    if [ -f "$RESULTS_CSV" ]; then
        echo
        echo "========== results header =========="
        head -n 1 "$RESULTS_CSV"

        echo
        echo "========== final epoch =========="
        tail -n 1 "$RESULTS_CSV"
    fi

    if [ -f "$BEST_WEIGHT" ]; then
        echo
        echo "========== best weight =========="
        ls -lh "$BEST_WEIGHT"
        sha256sum "$BEST_WEIGHT"
    fi

    if [ -f "$LAST_WEIGHT" ]; then
        echo
        echo "========== last weight =========="
        ls -lh "$LAST_WEIGHT"
        sha256sum "$LAST_WEIGHT"
    fi

    echo
    echo "exp02_6_yolo11n_baseline=$RESULT"

} > "$SUMMARY"

grep -nEi \
'Traceback|CUDA out of memory|RuntimeError|FileNotFoundError|AssertionError|Killed' \
"$LOG" \
> "$ABNORMAL" \
|| true

if [ ! -s "$ABNORMAL" ]; then
    echo "No abnormal messages detected." > "$ABNORMAL"
fi

echo
echo "========== final summary =========="
cat "$SUMMARY"

echo
echo "========== abnormal =========="
cat "$ABNORMAL"

echo
echo "log=$LOG"
echo "summary=$SUMMARY"
echo "environment=$ENVIRONMENT"
echo "abnormal=$ABNORMAL"

if [ "$RESULT" = "PASS" ]; then
    exit 0
fi

if [ "$TRAIN_RC" -ne 0 ]; then
    exit "$TRAIN_RC"
fi

exit 20
