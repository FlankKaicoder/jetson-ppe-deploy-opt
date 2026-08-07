#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${PPE_REPO_DIR:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PPE_JETSON_PYTHON:-/usr/bin/python3}"
onnx_path="/home/nvidia/models/jetson-ppe/exp06/yolo11n_baseline_exp06_b1_640_opset17.onnx"
probe_image="/home/nvidia/models/jetson-ppe/exp06/probe_train__image784.jpg"
calibration_root="/home/nvidia/models/jetson-ppe/exp08/calibration_20260807_145356"
manifest_path="$calibration_root/calibration/calibration_manifest.json"
artifact_root="/home/nvidia/models/jetson-ppe/exp08"
expected_onnx_sha256="305e23c65aa3b1d01e7b1a784c355665228f435f0b904b92ed2618954736d1f8"
expected_probe_sha256="39a263dd6931e7ca70b85348cdd35c3fed9ca5c938c391023d438b24fbe8910e"
expected_manifest_sha256="75b0c94f49aafc133402a43793dea40a7ca76131959b04043c87e69b44bd6d1d"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="exp08_1_int8_smoke_${timestamp}"
report_dir="$repo_dir/results/int8/$run_name"
artifact_dir="$artifact_root/$run_name"
engine_path="$artifact_dir/yolo11n_baseline_exp08_smoke_b1_640_int8.engine"
cache_path="$artifact_dir/yolo11n_baseline_exp08_smoke_int8.cache"

if [ -e "$report_dir" ] || [ -e "$artifact_dir" ]; then
    echo "ERROR: timestamp output already exists"
    exit 1
fi
mkdir -p "$report_dir/build" "$report_dir/execution" "$artifact_dir"
run_log="$report_dir/run.log"

fail_early() {
    local message="$1"
    echo "ERROR: $message" | tee -a "$run_log"
    printf '%s\n' 1 > "$report_dir/return_code.txt"
    printf 'result=FAIL\nreason=%s\n' "$message" > "$report_dir/failure_summary.txt"
    exit 1
}

[ -x "$python_bin" ] || fail_early "Python missing: $python_bin"
[ -s "$onnx_path" ] || fail_early "ONNX missing"
[ -s "$probe_image" ] || fail_early "probe image missing"
[ -s "$manifest_path" ] || fail_early "calibration manifest missing"
[ "$(sha256sum "$onnx_path" | cut -d ' ' -f 1)" = "$expected_onnx_sha256" ] || fail_early "ONNX SHA256 mismatch"
[ "$(sha256sum "$probe_image" | cut -d ' ' -f 1)" = "$expected_probe_sha256" ] || fail_early "probe SHA256 mismatch"
[ "$(sha256sum "$manifest_path" | cut -d ' ' -f 1)" = "$expected_manifest_sha256" ] || fail_early "manifest SHA256 mismatch"

cd "$repo_dir" || fail_early "cannot enter repository"
case "$(git branch --show-current)" in
    exp/08-*) ;;
    *) fail_early "unexpected Git branch: $(git branch --show-current)" ;;
esac

{
    echo "experiment=Exp08.1 INT8 smoke"
    echo "timestamp=$timestamp"
    echo "hostname=$(hostname)"
    echo "whoami=$(whoami)"
    echo "uname=$(uname -a)"
    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "python=$python_bin"
    "$python_bin" --version
    "$python_bin" -c 'import tensorrt as trt, torch, cv2; print("tensorrt=" + trt.__version__); print("torch=" + torch.__version__); print("opencv=" + cv2.__version__)'
    /usr/local/cuda/bin/nvcc --version
    nvpmodel -q
    echo "onnx_sha256=$expected_onnx_sha256"
    echo "manifest_sha256=$expected_manifest_sha256"
    echo "calibration_images=16_SMOKE_ONLY_NOT_FORMAL"
    echo "calibration_batch=1"
    echo "int8=true"
    echo "fp16_fallback=true"
    echo "tf32=false"
    echo "workspace_mib=1024"
    echo "builder_optimization_level=3"
} > "$report_dir/environment.txt" 2>&1

{
    echo "$python_bin tools/exp08_build_int8.py --limit 16 --batch-size 1"
    echo "$python_bin tools/exp08_int8_smoke.py --imgsz 640 --confidence 0.25 --nms-iou 0.70"
    echo "Smoke gate: build succeeds, output shape [1,7,8400], all outputs finite"
    echo "Raw/NMS deltas are diagnostic and do not replace formal full-test gates"
} > "$report_dir/command.txt"

"$python_bin" -u tools/exp08_build_int8.py \
    --onnx "$onnx_path" \
    --manifest "$manifest_path" \
    --engine "$engine_path" \
    --cache "$cache_path" \
    --report-dir "$report_dir/build" \
    --imgsz 640 \
    --batch-size 1 \
    --limit 16 \
    --workspace-mib 1024 \
    --builder-optimization-level 3 \
    2>&1 | tee "$run_log"
build_return_code=${PIPESTATUS[0]}
printf '%s\n' "$build_return_code" > "$report_dir/build_return_code.txt"

execution_return_code=99
if [ "$build_return_code" -eq 0 ] && [ -s "$engine_path" ] && [ -s "$cache_path" ]; then
    "$python_bin" -u tools/exp08_int8_smoke.py \
        --onnx "$onnx_path" \
        --engine "$engine_path" \
        --image "$probe_image" \
        --report-dir "$report_dir/execution" \
        --imgsz 640 \
        --confidence 0.25 \
        --nms-iou 0.70 \
        2>&1 | tee -a "$run_log"
    execution_return_code=${PIPESTATUS[0]}
fi
printf '%s\n' "$execution_return_code" > "$report_dir/execution_return_code.txt"

grep -nE 'Traceback|FATAL:|ERROR:|result=FAIL|"result": "FAIL"' "$run_log" > "$report_dir/abnormal.txt"
if [ "$build_return_code" -eq 0 ] && \
    [ "$execution_return_code" -eq 0 ] && \
    [ -s "$report_dir/build/summary.json" ] && \
    [ -s "$report_dir/execution/summary.json" ] && \
    grep -q '^result=PASS$' "$report_dir/build/summary.txt" && \
    grep -q '^result=PASS$' "$report_dir/execution/summary.txt" && \
    [ ! -s "$report_dir/abnormal.txt" ]; then
    final_result="PASS"
    final_code=0
else
    final_result="FAIL"
    final_code=1
fi

{
    echo "result=$final_result"
    echo "build_return_code=$build_return_code"
    echo "execution_return_code=$execution_return_code"
    echo "report_dir=$report_dir"
    echo "artifact_dir=$artifact_dir"
    if [ -s "$engine_path" ]; then
        echo "engine_sha256=$(sha256sum "$engine_path" | cut -d ' ' -f 1)"
    fi
    if [ -s "$cache_path" ]; then
        echo "cache_sha256=$(sha256sum "$cache_path" | cut -d ' ' -f 1)"
    fi
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

echo "exp08_1_int8_smoke=PASS"
exit 0
