#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 file-smoke|file-formal|camera-smoke|camera-formal [video_path]" >&2
  exit 2
fi

MODE="$1"
VIDEO_PATH="${2:-/home/nvidia/imx219_test_sensor0.mp4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${REPO_ROOT}/build/exp11/exp11_video_infer"
ENGINE="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

case "${MODE}" in
  file-smoke)
    RUN_ID="exp11_0_file_video_smoke_${TIMESTAMP}"
    SOURCE_ARGS=(--source-type file --source "${VIDEO_PATH}" --max-frames 5)
    ;;
  file-formal)
    RUN_ID="exp11_1_file_video_formal_${TIMESTAMP}"
    SOURCE_ARGS=(--source-type file --source "${VIDEO_PATH}" --max-frames 0)
    ;;
  camera-smoke)
    RUN_ID="exp11_2_imx219_smoke_${TIMESTAMP}"
    SOURCE_ARGS=(--source-type camera --sensor-id 0 --width 1920 --height 1080 --fps 30 --max-frames 30)
    ;;
  camera-formal)
    RUN_ID="exp11_3_imx219_formal_${TIMESTAMP}"
    SOURCE_ARGS=(--source-type camera --sensor-id 0 --width 1920 --height 1080 --fps 30 --max-frames 300)
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${REPO_ROOT}/results/video/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
for path in "${BINARY}" "${ENGINE}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'result=FAIL missing_input=%s\n' "${path}" \
      | tee "${OUTPUT_DIR}/failure_summary.txt"
    exit 1
  fi
done
if [[ "${MODE}" == file-* && ! -f "${VIDEO_PATH}" ]]; then
  printf 'result=FAIL missing_video=%s\n' "${VIDEO_PATH}" \
    | tee "${OUTPUT_DIR}/failure_summary.txt"
  exit 1
fi

ENGINE_SHA="$(sha256sum "${ENGINE}" | awk '{print $1}')"
if [[ "${ENGINE_SHA}" != "88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83" ]]; then
  printf 'result=FAIL engine_sha256=%s\n' "${ENGINE_SHA}" \
    | tee "${OUTPUT_DIR}/failure_summary.txt"
  exit 1
fi

{
  hostname
  uname -a
  git -C "${REPO_ROOT}" rev-parse HEAD
  nvpmodel -q
  printf 'engine_sha256=%s\n' "${ENGINE_SHA}"
  printf 'mode=%s\n' "${MODE}"
} > "${OUTPUT_DIR}/environment.txt"

COMMAND=("${BINARY}" --engine "${ENGINE}" "${SOURCE_ARGS[@]}"
  --output-dir "${OUTPUT_DIR}" --warmup 2 --confidence 0.25 --nms-iou 0.70)
printf '%q ' "${COMMAND[@]}" > "${OUTPUT_DIR}/command.txt"
printf '\n' >> "${OUTPUT_DIR}/command.txt"
"${COMMAND[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
RUN_RC=${PIPESTATUS[0]}
printf '%s\n' "${RUN_RC}" > "${OUTPUT_DIR}/return_code.txt"
if [[ ${RUN_RC} -ne 0 ]]; then
  printf 'result=FAIL mode=%s return_code=%s\n' "${MODE}" "${RUN_RC}" \
    > "${OUTPUT_DIR}/failure_summary.txt"
  exit "${RUN_RC}"
fi

for output in summary.json frames.csv detections.csv first_annotated.jpg last_annotated.jpg; do
  if [[ ! -s "${OUTPUT_DIR}/${output}" ]]; then
    printf 'result=FAIL missing_output=%s\n' "${output}" \
      > "${OUTPUT_DIR}/failure_summary.txt"
    exit 1
  fi
done
sha256sum "${OUTPUT_DIR}/detections.csv" \
  "${OUTPUT_DIR}/first_annotated.jpg" "${OUTPUT_DIR}/last_annotated.jpg" \
  > "${OUTPUT_DIR}/sha256.txt"
printf 'result=PASS mode=%s output_dir=%s\n' "${MODE}" "${OUTPUT_DIR}" \
  | tee "${OUTPUT_DIR}/summary.txt"
