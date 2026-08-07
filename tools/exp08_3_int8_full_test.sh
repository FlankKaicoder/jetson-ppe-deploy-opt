#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${PPE_REPO_DIR:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PPE_JETSON_PYTHON:-/usr/bin/python3}"
fp16_engine="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
int8_engine="/home/nvidia/models/jetson-ppe/exp08/exp08_2_int8_formal_20260807_153244/yolo11n_baseline_exp08_b1_640_int8.engine"
data_yaml="/home/nvidia/datasets/jetson-ppe-exp07/construction_ppe3_test_v1/construction_ppe3_jetson_test.yaml"
dataset_archive="/home/nvidia/datasets/jetson-ppe-exp07/construction_ppe3_test_split.tar.gz"
exp07_reference="$repo_dir/results/tensorrt/exp07_1b_full_test_consistency_20260806_175921/summary.json"
expected_fp16_sha256="88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83"
expected_int8_sha256="5787fb3bae4dbd00909c1762efc9263566044bc4dc35a836c950312e85895f26"
expected_dataset_sha256="3bf3addcb79e7ac46163f7a294265a92c5f84c7e633f56da0b16e22f33400f4a"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="exp08_3_int8_full_test_${timestamp}"
report_dir="$repo_dir/results/int8/$run_name"

if [ -e "$report_dir" ]; then
    echo "ERROR: report directory already exists"
    exit 1
fi
mkdir -p "$report_dir/backend_metrics" "$report_dir/val_runs"
run_log="$report_dir/run.log"

fail_early() {
    local message="$1"
    echo "ERROR: $message" | tee -a "$run_log"
    printf '%s\n' 1 > "$report_dir/return_code.txt"
    printf 'result=FAIL\nreason=%s\n' "$message" > "$report_dir/failure_summary.txt"
    exit 1
}

[ -x "$python_bin" ] || fail_early "Python missing"
[ -s "$fp16_engine" ] || fail_early "FP16 engine missing"
[ -s "$int8_engine" ] || fail_early "INT8 engine missing"
[ -s "$data_yaml" ] || fail_early "test YAML missing"
[ -s "$dataset_archive" ] || fail_early "test archive missing"
[ -s "$exp07_reference" ] || fail_early "Exp07 reference missing"
[ "$(sha256sum "$fp16_engine" | cut -d ' ' -f 1)" = "$expected_fp16_sha256" ] || fail_early "FP16 SHA256 mismatch"
[ "$(sha256sum "$int8_engine" | cut -d ' ' -f 1)" = "$expected_int8_sha256" ] || fail_early "INT8 SHA256 mismatch"
[ "$(sha256sum "$dataset_archive" | cut -d ' ' -f 1)" = "$expected_dataset_sha256" ] || fail_early "dataset SHA256 mismatch"

cd "$repo_dir" || fail_early "cannot enter repository"
case "$(git branch --show-current)" in
    exp/08-*) ;;
    *) fail_early "unexpected Git branch" ;;
esac

{
    echo "experiment=Exp08.3 INT8 full test and scale audit"
    echo "timestamp=$timestamp"
    echo "hostname=$(hostname)"
    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "fp16_sha256=$expected_fp16_sha256"
    echo "int8_sha256=$expected_int8_sha256"
    echo "dataset_sha256=$expected_dataset_sha256"
    echo "test_images=219"
    echo "test_instances=840"
    echo "imgsz=640"
    echo "batch=1"
    echo "rect=false"
    echo "map50_95_max_drop=0.010"
    echo "map50_max_drop=0.015"
    echo "tiny_small_recall_max_drop=0.050"
    "$python_bin" -m pip show ultralytics tensorrt torch
} > "$report_dir/environment.txt" 2>&1

{
    echo "Separate backend processes; same Jetson runtime"
    echo "AP: split=test imgsz=640 batch=1 rect=false workers=2"
    echo "Scale: conf=0.25 NMS_IoU=0.70 match_IoU=0.50 max_visuals=0"
    echo "Frozen gates: mAP50-95 -0.010, mAP50 -0.015, tiny+small recall -0.050"
} > "$report_dir/command.txt"

run_backend() {
    local name="$1"
    local engine="$2"
    YOLO_AUTOINSTALL=false "$python_bin" -u tools/exp07_eval_backend.py \
        --model "$engine" \
        --data "$data_yaml" \
        --output "$report_dir/backend_metrics/${name}.json" \
        --project "$report_dir/val_runs" \
        --name "$name" \
        --device 0 \
        --imgsz 640 \
        --batch 1 \
        --workers 2 \
        2>&1 | tee -a "$run_log"
    return ${PIPESTATUS[0]}
}

run_scale() {
    local name="$1"
    local engine="$2"
    YOLO_AUTOINSTALL=false "$python_bin" -u tools/exp02_8_baseline_error_size_audit.py \
        --weights "$engine" \
        --data "$data_yaml" \
        --output-dir "$report_dir/${name}_scale" \
        --imgsz 640 \
        --batch 1 \
        --conf 0.25 \
        --nms-iou 0.70 \
        --match-iou 0.50 \
        --max-visuals 0 \
        2>&1 | tee -a "$run_log"
    return ${PIPESTATUS[0]}
}

run_backend fp16 "$fp16_engine"
fp16_backend_return_code=$?
run_backend int8 "$int8_engine"
int8_backend_return_code=$?
run_scale fp16 "$fp16_engine"
fp16_scale_return_code=$?
run_scale int8 "$int8_engine"
int8_scale_return_code=$?

printf '%s\n' "$fp16_backend_return_code" > "$report_dir/fp16_backend_return_code.txt"
printf '%s\n' "$int8_backend_return_code" > "$report_dir/int8_backend_return_code.txt"
printf '%s\n' "$fp16_scale_return_code" > "$report_dir/fp16_scale_return_code.txt"
printf '%s\n' "$int8_scale_return_code" > "$report_dir/int8_scale_return_code.txt"

collector_return_code=99
if [ "$fp16_backend_return_code" -eq 0 ] && \
    [ "$int8_backend_return_code" -eq 0 ] && \
    [ "$fp16_scale_return_code" -eq 0 ] && \
    [ "$int8_scale_return_code" -eq 0 ]; then
    "$python_bin" -u tools/exp08_collect_full_test.py \
        --report-dir "$report_dir" \
        --exp07-reference "$exp07_reference" \
        --fp16-metrics "$report_dir/backend_metrics/fp16.json" \
        --int8-metrics "$report_dir/backend_metrics/int8.json" \
        --fp16-scale "$report_dir/fp16_scale/summary.json" \
        --int8-scale "$report_dir/int8_scale/summary.json" \
        --fp16-engine "$fp16_engine" \
        --int8-engine "$int8_engine" \
        --map50-95-max-drop 0.010 \
        --map50-max-drop 0.015 \
        --tiny-small-max-drop 0.050 \
        2>&1 | tee -a "$run_log"
    collector_return_code=${PIPESTATUS[0]}
fi
printf '%s\n' "$collector_return_code" > "$report_dir/collector_return_code.txt"

grep -nE 'Traceback|FATAL:|ERROR:|result=FAIL|"result": "FAIL"' "$run_log" > "$report_dir/abnormal.txt"
if [ "$collector_return_code" -eq 0 ] && \
    [ -s "$report_dir/summary.json" ] && \
    grep -q '^result=PASS$' "$report_dir/summary.txt" && \
    [ ! -s "$report_dir/abnormal.txt" ]; then
    final_result="PASS"
    final_code=0
else
    final_result="FAIL"
    final_code=1
fi

{
    echo "result=$final_result"
    echo "fp16_backend_return_code=$fp16_backend_return_code"
    echo "int8_backend_return_code=$int8_backend_return_code"
    echo "fp16_scale_return_code=$fp16_scale_return_code"
    echo "int8_scale_return_code=$int8_scale_return_code"
    echo "collector_return_code=$collector_return_code"
    echo "report_dir=$report_dir"
} > "$report_dir/runner_summary.txt"
printf '%s\n' "$final_code" > "$report_dir/return_code.txt"

if [ "$final_code" -ne 0 ]; then
    {
        cat "$report_dir/runner_summary.txt"
        echo "last_log_lines:"
        tr '\r' '\n' < "$run_log" | tail -n 160
    } > "$report_dir/failure_summary.txt"
    exit "$final_code"
fi

echo "exp08_3_int8_full_test=PASS"
exit 0
