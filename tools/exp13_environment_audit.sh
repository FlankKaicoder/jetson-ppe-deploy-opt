#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${REPO_ROOT}/results/profiling/exp13_0_environment_${TIMESTAMP}"
ENGINE="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
VIDEO="/home/nvidia/imx219_test_sensor0.mp4"
ENGINE_SHA="88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83"
VIDEO_SHA="f00e116a9d8f1a9f2bdcb4fc08c8720019e6f61047934613b0940da961a66665"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee "${OUTPUT_DIR}/run.log") 2>&1

result=0
for path in "${ENGINE}" "${VIDEO}" "${REPO_ROOT}/tools/exp12_clock_status.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing_input=${path}"
    result=1
  fi
done
for tool in nsys python3 cmake g++ /usr/local/cuda/bin/nvcc /usr/src/tensorrt/bin/trtexec; do
  if ! command -v "${tool}" >/dev/null 2>&1 && [[ ! -x "${tool}" ]]; then
    echo "missing_tool=${tool}"
    result=1
  fi
done
if [[ ${result} -ne 0 ]]; then
  printf '%s\n' "${result}" > "${OUTPUT_DIR}/return_code.txt"
  printf 'result=FAIL stage=input_check\n' > "${OUTPUT_DIR}/failure_summary.txt"
  exit "${result}"
fi

{
  hostname
  whoami
  pwd
  uname -a
  git -C "${REPO_ROOT}" rev-parse HEAD
  git -C "${REPO_ROOT}" branch --show-current
  git -C "${REPO_ROOT}" status --short
  nvpmodel -q
  nsys --version
  /usr/local/cuda/bin/nvcc --version
  dpkg-query -W nvinfer-bin libnvinfer10 2>/dev/null
  cmake --version
  g++ --version
  df -h "${REPO_ROOT}"
} > "${OUTPUT_DIR}/environment.txt" 2>&1

sha256sum "${ENGINE}" "${VIDEO}" > "${OUTPUT_DIR}/input_sha256.txt"
actual_engine_sha="$(sha256sum "${ENGINE}" | awk '{print $1}')"
actual_video_sha="$(sha256sum "${VIDEO}" | awk '{print $1}')"
if [[ "${actual_engine_sha}" != "${ENGINE_SHA}" ||
      "${actual_video_sha}" != "${VIDEO_SHA}" ]]; then
  printf 'result=FAIL engine_sha=%s video_sha=%s\n' \
    "${actual_engine_sha}" "${actual_video_sha}" \
    > "${OUTPUT_DIR}/failure_summary.txt"
  printf '1\n' > "${OUTPUT_DIR}/return_code.txt"
  exit 1
fi

python3 "${REPO_ROOT}/tools/exp12_clock_status.py" \
  --output "${OUTPUT_DIR}/clock_status.json"
nsys profile --help > "${OUTPUT_DIR}/nsys_profile_help.txt" 2>&1
nsys stats --help > "${OUTPUT_DIR}/nsys_stats_help.txt" 2>&1
/usr/src/tensorrt/bin/trtexec --help \
  > "${OUTPUT_DIR}/trtexec_help.txt" 2>&1
printf '0\n' > "${OUTPUT_DIR}/return_code.txt"
printf 'result=PASS output_dir=%s\n' "${OUTPUT_DIR}" \
  | tee "${OUTPUT_DIR}/summary.txt"

