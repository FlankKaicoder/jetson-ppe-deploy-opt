#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 benchmark|profile baseline|A|B|C file|camera frame_count" >&2
  exit 2
fi

RUN_KIND="$1"
VARIANT="$2"
SOURCE_TYPE="$3"
FRAME_COUNT="$4"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
VIDEO="/home/nvidia/imx219_test_sensor0.mp4"
ENGINE_SHA="88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83"
VIDEO_SHA="f00e116a9d8f1a9f2bdcb4fc08c8720019e6f61047934613b0940da961a66665"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [[ "${RUN_KIND}" != "benchmark" && "${RUN_KIND}" != "profile" ]]; then
  echo "RUN_KIND must be benchmark or profile" >&2
  exit 2
fi
if [[ "${VARIANT}" != "baseline" && "${VARIANT}" != "A" &&
      "${VARIANT}" != "B" && "${VARIANT}" != "C" ]]; then
  echo "VARIANT must be baseline, A, B, or C" >&2
  exit 2
fi
if [[ "${SOURCE_TYPE}" != "file" && "${SOURCE_TYPE}" != "camera" ]]; then
  echo "SOURCE_TYPE must be file or camera" >&2
  exit 2
fi
if ! [[ "${FRAME_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "frame_count must be a positive integer" >&2
  exit 2
fi

if [[ "${RUN_KIND}" == "profile" ]]; then
  BUILD_DIR="${REPO_ROOT}/build/exp14_nvtx"
else
  BUILD_DIR="${REPO_ROOT}/build/exp14_plain"
fi
if [[ "${VARIANT}" == "baseline" ]]; then
  if [[ "${RUN_KIND}" == "profile" ]]; then
    BINARY="${BUILD_DIR}/exp13_profiled_video_infer"
  else
    BINARY="${BUILD_DIR}/exp11_video_infer"
  fi
else
  BINARY="${BUILD_DIR}/exp14_async_pipeline"
fi

RUN_ID="exp14_${RUN_KIND}_${VARIANT}_${SOURCE_TYPE}_${TIMESTAMP}"
OUTPUT_DIR="${REPO_ROOT}/results/pipeline/${RUN_ID}"
APP_OUTPUT="${OUTPUT_DIR}/app_output"
mkdir -p "${APP_OUTPUT}"

for path in "${BINARY}" "${ENGINE}" "${REPO_ROOT}/tools/exp12_clock_status.py"; do
  if [[ ! -e "${path}" ]]; then
    printf 'result=FAIL missing_input=%s\n' "${path}" \
      > "${OUTPUT_DIR}/failure_summary.txt"
    exit 1
  fi
done
if [[ "${SOURCE_TYPE}" == "file" && ! -f "${VIDEO}" ]]; then
  printf 'result=FAIL missing_video=%s\n' "${VIDEO}" \
    > "${OUTPUT_DIR}/failure_summary.txt"
  exit 1
fi
if [[ "$(sha256sum "${ENGINE}" | awk '{print $1}')" != "${ENGINE_SHA}" ]]; then
  printf 'result=FAIL engine_sha256_mismatch\n' \
    > "${OUTPUT_DIR}/failure_summary.txt"
  exit 1
fi
if [[ "${SOURCE_TYPE}" == "file" &&
      "$(sha256sum "${VIDEO}" | awk '{print $1}')" != "${VIDEO_SHA}" ]]; then
  printf 'result=FAIL video_sha256_mismatch\n' \
    > "${OUTPUT_DIR}/failure_summary.txt"
  exit 1
fi

if [[ "${SOURCE_TYPE}" == "file" ]]; then
  SOURCE_ARGS=(--source-type file --source "${VIDEO}"
    --max-frames "${FRAME_COUNT}")
else
  SOURCE_ARGS=(--source-type camera --sensor-id 0 --width 1920 --height 1080
    --fps 30 --max-frames "${FRAME_COUNT}")
fi
APP_COMMAND=("${BINARY}" --engine "${ENGINE}" "${SOURCE_ARGS[@]}"
  --output-dir "${APP_OUTPUT}" --warmup 2 --confidence 0.25 --nms-iou 0.70)
if [[ "${VARIANT}" != "baseline" ]]; then
  APP_COMMAND+=(--variant "${VARIANT}")
fi

{
  hostname
  whoami
  uname -a
  git -C "${REPO_ROOT}" rev-parse HEAD
  git -C "${REPO_ROOT}" branch --show-current
  nvpmodel -q
  printf 'run_kind=%s\nvariant=%s\nsource_type=%s\nframe_count=%s\n' \
    "${RUN_KIND}" "${VARIANT}" "${SOURCE_TYPE}" "${FRAME_COUNT}"
  sha256sum "${BINARY}" "${ENGINE}"
  if [[ "${SOURCE_TYPE}" == "file" ]]; then
    sha256sum "${VIDEO}"
  fi
} > "${OUTPUT_DIR}/environment.txt" 2>&1
python3 "${REPO_ROOT}/tools/exp12_clock_status.py" \
  --output "${OUTPUT_DIR}/clock_status.json" >/dev/null
printf '%q ' "${APP_COMMAND[@]}" > "${OUTPUT_DIR}/app_command.txt"
printf '\n' >> "${OUTPUT_DIR}/app_command.txt"

start_ns="$(date +%s%N)"
if [[ "${RUN_KIND}" == "profile" ]]; then
  PROFILE_COMMAND=(nsys profile --trace=cuda,nvtx,osrt --sample=none
    --cpuctxsw=none --force-overwrite=true
    --output="${OUTPUT_DIR}/report" "${APP_COMMAND[@]}")
  printf '%q ' "${PROFILE_COMMAND[@]}" > "${OUTPUT_DIR}/command.txt"
  printf '\n' >> "${OUTPUT_DIR}/command.txt"
  "${PROFILE_COMMAND[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
else
  printf '%q ' "${APP_COMMAND[@]}" > "${OUTPUT_DIR}/command.txt"
  printf '\n' >> "${OUTPUT_DIR}/command.txt"
  "${APP_COMMAND[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
fi
run_rc=${PIPESTATUS[0]}
end_ns="$(date +%s%N)"
printf '%s\n' "${start_ns}" > "${OUTPUT_DIR}/wall_start_ns.txt"
printf '%s\n' "${end_ns}" > "${OUTPUT_DIR}/wall_end_ns.txt"
printf '%s\n' "${run_rc}" > "${OUTPUT_DIR}/return_code.txt"
if [[ ${run_rc} -ne 0 ]]; then
  printf 'result=FAIL run_kind=%s variant=%s source_type=%s return_code=%s\n' \
    "${RUN_KIND}" "${VARIANT}" "${SOURCE_TYPE}" "${run_rc}" \
    > "${OUTPUT_DIR}/failure_summary.txt"
  exit "${run_rc}"
fi

for output in summary.json frames.csv detections.csv; do
  if [[ ! -s "${APP_OUTPUT}/${output}" ]]; then
    printf 'result=FAIL missing_output=%s\n' "${output}" \
      > "${OUTPUT_DIR}/failure_summary.txt"
    exit 1
  fi
done

VALIDATE_COMMAND=(python3 "${REPO_ROOT}/tools/exp14_validate.py" "${OUTPUT_DIR}"
  --expected-frames "${FRAME_COUNT}")
if [[ "${SOURCE_TYPE}" == "file" && "${FRAME_COUNT}" == "150" ]]; then
  VALIDATE_COMMAND+=(--expected-detections 151
    --expected-detection-sha256
    9f3f33459f8d086a74249a57f21f158a73ca794a2229a9e1af40a03de34e2d8a)
fi
if [[ "${VARIANT}" != "baseline" ]]; then
  VALIDATE_COMMAND+=(--expected-variant "${VARIANT}")
fi
"${VALIDATE_COMMAND[@]}" > "${OUTPUT_DIR}/validation.log" 2>&1
validation_rc=$?
printf '%s\n' "${validation_rc}" > "${OUTPUT_DIR}/validation_return_code.txt"
if [[ ${validation_rc} -ne 0 ]]; then
  printf 'result=FAIL validation_rc=%s\n' "${validation_rc}" \
    > "${OUTPUT_DIR}/failure_summary.txt"
  exit 1
fi

if [[ "${RUN_KIND}" == "profile" ]]; then
  if [[ ! -s "${OUTPUT_DIR}/report.nsys-rep" ]]; then
    printf 'result=FAIL missing_report\n' > "${OUTPUT_DIR}/failure_summary.txt"
    exit 1
  fi
  nsys export --type=sqlite --force-overwrite=true \
    --output="${OUTPUT_DIR}/report.sqlite" \
    "${OUTPUT_DIR}/report.nsys-rep" > "${OUTPUT_DIR}/export.log" 2>&1
  export_rc=$?
  nsys stats --force-export=true --sqlite "${OUTPUT_DIR}/report.sqlite" \
    "${OUTPUT_DIR}/report.nsys-rep" > "${OUTPUT_DIR}/nsys_stats.txt" 2>&1
  stats_rc=$?
  printf '%s\n' "${export_rc}" > "${OUTPUT_DIR}/export_return_code.txt"
  printf '%s\n' "${stats_rc}" > "${OUTPUT_DIR}/stats_return_code.txt"
  if [[ ${export_rc} -ne 0 || ${stats_rc} -ne 0 ]]; then
    printf 'result=FAIL export_rc=%s stats_rc=%s\n' \
      "${export_rc}" "${stats_rc}" > "${OUTPUT_DIR}/failure_summary.txt"
    exit 1
  fi
  if [[ "${VARIANT}" == "C" ]]; then
    python3 "${REPO_ROOT}/tools/exp14_analyze_nsys.py" \
      "${OUTPUT_DIR}/report.sqlite" --output-dir "${OUTPUT_DIR}" \
      --expected-frames "${FRAME_COUNT}" --warmup 2 \
      --source-type "${SOURCE_TYPE}" > "${OUTPUT_DIR}/analyze.log" 2>&1
    analyze_rc=$?
    printf '%s\n' "${analyze_rc}" > "${OUTPUT_DIR}/analyze_return_code.txt"
    if [[ ${analyze_rc} -ne 0 ]]; then
      printf 'result=FAIL analyze_rc=%s\n' "${analyze_rc}" \
        > "${OUTPUT_DIR}/failure_summary.txt"
      exit 1
    fi
  fi
fi

sha256sum "${APP_OUTPUT}/summary.json" "${APP_OUTPUT}/frames.csv" \
  "${APP_OUTPUT}/detections.csv" "${OUTPUT_DIR}/validation.json" \
  > "${OUTPUT_DIR}/app_sha256.txt"
if [[ "${RUN_KIND}" == "profile" ]]; then
  sha256sum "${OUTPUT_DIR}/report.nsys-rep" "${OUTPUT_DIR}/report.sqlite" \
    >> "${OUTPUT_DIR}/app_sha256.txt"
  if [[ -s "${OUTPUT_DIR}/exp14_timeline_summary.json" ]]; then
    sha256sum "${OUTPUT_DIR}/exp14_timeline_summary.json" \
      >> "${OUTPUT_DIR}/app_sha256.txt"
  fi
fi
printf 'result=PASS run_id=%s output_dir=%s\n' "${RUN_ID}" "${OUTPUT_DIR}" \
  | tee "${OUTPUT_DIR}/summary.txt"
