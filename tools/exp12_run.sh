#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 smoke|baseline|performance|stability" >&2
  exit 2
fi

MODE="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/usr/bin/python3"
BINARY="${REPO_ROOT}/build/exp11/exp11_video_infer"
ENGINE="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
ENGINE_SHA="88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83"
BINARY_SHA="bf3717d8b4feb17617ea4e831dcc6fffdb69a13d52449c0727aa8697ada4a5c0"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

case "${MODE}" in
  smoke)
    RUN_ID="exp12_0_monitor_smoke_${TIMESTAMP}"
    MAX_FRAMES=300
    WARMUP=2
    ;;
  baseline)
    RUN_ID="exp12_1_unlocked_baseline_${TIMESTAMP}"
    MAX_FRAMES=1800
    WARMUP=20
    ;;
  performance)
    RUN_ID="exp12_2_locked_performance_${TIMESTAMP}"
    MAX_FRAMES=1800
    WARMUP=20
    ;;
  stability)
    RUN_ID="exp12_3_locked_stability_${TIMESTAMP}"
    MAX_FRAMES=54000
    WARMUP=20
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${REPO_ROOT}/results/benchmark/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

fail_early() {
  local message="$1"
  echo "result=FAIL reason=${message}" | tee "${OUTPUT_DIR}/failure_summary.txt"
  echo 1 > "${OUTPUT_DIR}/return_code.txt"
  exit 1
}

[[ -x "${PYTHON_BIN}" ]] || fail_early "python_missing"
[[ -x "${BINARY}" ]] || fail_early "binary_missing"
[[ -s "${ENGINE}" ]] || fail_early "engine_missing"
[[ "$(sha256sum "${ENGINE}" | awk '{print $1}')" == "${ENGINE_SHA}" ]] || fail_early "engine_sha_mismatch"
[[ "$(sha256sum "${BINARY}" | awk '{print $1}')" == "${BINARY_SHA}" ]] || fail_early "binary_sha_mismatch"
[[ "$(git -C "${REPO_ROOT}" branch --show-current)" == exp/12-* ]] || fail_early "unexpected_git_branch"
nvpmodel -q | grep -q "NV Power Mode: 25W" || fail_early "nvpmodel_not_25w"

CLOCK_ARGS=()
if [[ "${MODE}" == "performance" || "${MODE}" == "stability" ]]; then
  CLOCK_ARGS=(--require-locked)
fi
"${PYTHON_BIN}" "${REPO_ROOT}/tools/exp12_clock_status.py" \
  --output "${OUTPUT_DIR}/clock_status.json" "${CLOCK_ARGS[@]}" \
  > "${OUTPUT_DIR}/clock_status_stdout.txt" 2>&1
CLOCK_RC=$?
[[ ${CLOCK_RC} -eq 0 ]] || fail_early "clock_status_failed"

{
  echo "experiment=Exp12"
  echo "mode=${MODE}"
  echo "timestamp=${TIMESTAMP}"
  echo "hostname=$(hostname)"
  echo "git_branch=$(git -C "${REPO_ROOT}" branch --show-current)"
  echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo "engine_sha256=${ENGINE_SHA}"
  echo "binary_sha256=${BINARY_SHA}"
  echo "max_frames=${MAX_FRAMES}"
  echo "warmup=${WARMUP}"
  echo "sample_interval_seconds=1"
  echo "timing_scope=capture+H2D+CUDA_preprocess+TensorRT+D2H+NMS"
  nvpmodel -q
  /usr/local/cuda/bin/nvcc --version | tail -1
  dpkg-query -W libnvinfer10 2>/dev/null
  pkg-config --modversion opencv4 gstreamer-1.0
} > "${OUTPUT_DIR}/environment.txt"

APP_COMMAND=("${BINARY}" --engine "${ENGINE}" --source-type camera
  --sensor-id 0 --width 1920 --height 1080 --fps 30
  --max-frames "${MAX_FRAMES}" --output-dir "${OUTPUT_DIR}"
  --warmup "${WARMUP}" --confidence 0.25 --nms-iou 0.70)
printf '%q ' "${APP_COMMAND[@]}" > "${OUTPUT_DIR}/command.txt"
printf '\n' >> "${OUTPUT_DIR}/command.txt"

"${PYTHON_BIN}" "${REPO_ROOT}/tools/exp12_monitor.py" \
  --output-dir "${OUTPUT_DIR}" --interval-seconds 1 \
  --temperature-stop-c 90 -- "${APP_COMMAND[@]}"
MONITOR_RC=$?
echo "${MONITOR_RC}" > "${OUTPUT_DIR}/monitor_return_code.txt"
if [[ ${MONITOR_RC} -ne 0 ]]; then
  echo "result=FAIL stage=monitor return_code=${MONITOR_RC}" \
    > "${OUTPUT_DIR}/failure_summary.txt"
  echo "${MONITOR_RC}" > "${OUTPUT_DIR}/return_code.txt"
  exit "${MONITOR_RC}"
fi

"${PYTHON_BIN}" "${REPO_ROOT}/tools/exp12_analyze.py" \
  --run-dir "${OUTPUT_DIR}" --mode "${MODE}" \
  > "${OUTPUT_DIR}/analyze_stdout.txt" 2>&1
ANALYZE_RC=$?
echo "${ANALYZE_RC}" > "${OUTPUT_DIR}/analyze_return_code.txt"
echo "${ANALYZE_RC}" > "${OUTPUT_DIR}/return_code.txt"
if [[ ${ANALYZE_RC} -ne 0 ]]; then
  cp "${OUTPUT_DIR}/exp12_summary.txt" "${OUTPUT_DIR}/failure_summary.txt"
  exit "${ANALYZE_RC}"
fi

sha256sum "${OUTPUT_DIR}/summary.json" "${OUTPUT_DIR}/frames.csv" \
  "${OUTPUT_DIR}/detections.csv" "${OUTPUT_DIR}/tegrastats.log" \
  "${OUTPUT_DIR}/process_samples.csv" "${OUTPUT_DIR}/exp12_summary.json" \
  > "${OUTPUT_DIR}/sha256.txt"
echo "result=PASS mode=${MODE} output_dir=${OUTPUT_DIR}" | tee "${OUTPUT_DIR}/runner_summary.txt"
