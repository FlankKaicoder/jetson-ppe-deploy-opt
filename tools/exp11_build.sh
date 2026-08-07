#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${REPO_ROOT}/results/video/exp11_build_${TIMESTAMP}"
BUILD_DIR="${REPO_ROOT}/build/exp11"
mkdir -p "${OUTPUT_DIR}"

for path in \
  "${REPO_ROOT}/app/CMakeLists.txt" \
  "${REPO_ROOT}/app/src/exp11_video_infer.cpp" \
  "/usr/local/cuda/bin/nvcc"; do
  if [[ ! -e "${path}" ]]; then
    printf 'missing_input=%s\n' "${path}" | tee "${OUTPUT_DIR}/failure_summary.txt"
    exit 1
  fi
done

{
  hostname
  uname -a
  /usr/local/cuda/bin/nvcc --version
  cmake --version
  g++ --version | head -1
  pkg-config --modversion opencv4 gstreamer-1.0 gstreamer-app-1.0
  git -C "${REPO_ROOT}" rev-parse HEAD
} > "${OUTPUT_DIR}/environment.txt"

printf 'cmake -S %q -B %q -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc\n' \
  "${REPO_ROOT}/app" "${BUILD_DIR}" > "${OUTPUT_DIR}/command.txt"
cmake -S "${REPO_ROOT}/app" -B "${BUILD_DIR}" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  2>&1 | tee "${OUTPUT_DIR}/configure.log"
CONFIGURE_RC=${PIPESTATUS[0]}
if [[ ${CONFIGURE_RC} -ne 0 ]]; then
  printf '%s\n' "${CONFIGURE_RC}" > "${OUTPUT_DIR}/return_code.txt"
  printf 'result=FAIL stage=configure return_code=%s\n' "${CONFIGURE_RC}" \
    > "${OUTPUT_DIR}/failure_summary.txt"
  exit "${CONFIGURE_RC}"
fi

cmake --build "${BUILD_DIR}" --parallel 4 \
  2>&1 | tee "${OUTPUT_DIR}/build.log"
BUILD_RC=${PIPESTATUS[0]}
printf '%s\n' "${BUILD_RC}" > "${OUTPUT_DIR}/return_code.txt"
if [[ ${BUILD_RC} -ne 0 ]]; then
  printf 'result=FAIL stage=build return_code=%s\n' "${BUILD_RC}" \
    > "${OUTPUT_DIR}/failure_summary.txt"
  exit "${BUILD_RC}"
fi

BINARY="${BUILD_DIR}/exp11_video_infer"
if [[ ! -x "${BINARY}" ]]; then
  printf 'result=FAIL missing_binary=%s\n' "${BINARY}" \
    > "${OUTPUT_DIR}/failure_summary.txt"
  exit 1
fi
sha256sum "${BINARY}" > "${OUTPUT_DIR}/sha256.txt"
printf 'result=PASS binary=%s output_dir=%s\n' "${BINARY}" "${OUTPUT_DIR}" \
  | tee "${OUTPUT_DIR}/summary.txt"
