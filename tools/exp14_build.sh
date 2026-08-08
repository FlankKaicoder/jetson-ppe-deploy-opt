#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${REPO_ROOT}/results/pipeline/exp14_0_build_${TIMESTAMP}"
PLAIN_BUILD="${REPO_ROOT}/build/exp14_plain"
NVTX_BUILD="${REPO_ROOT}/build/exp14_nvtx"
mkdir -p "${OUTPUT_DIR}"

for path in \
  "${REPO_ROOT}/app/CMakeLists.txt" \
  "${REPO_ROOT}/app/src/exp11_video_infer.cpp" \
  "${REPO_ROOT}/app/src/exp14_async_pipeline.cpp" \
  "${REPO_ROOT}/cuda/src/cuda_preprocess.cu" \
  "${REPO_ROOT}/runtime/src/trt_runtime.cpp" \
  "/usr/local/cuda/bin/nvcc"; do
  if [[ ! -e "${path}" ]]; then
    printf 'result=FAIL missing_input=%s\n' "${path}" \
      > "${OUTPUT_DIR}/failure_summary.txt"
    exit 1
  fi
done

{
  hostname
  whoami
  uname -a
  git -C "${REPO_ROOT}" rev-parse HEAD
  git -C "${REPO_ROOT}" branch --show-current
  /usr/local/cuda/bin/nvcc --version
  cmake --version
  g++ --version | head -n 1
  pkg-config --modversion opencv4 gstreamer-1.0
} > "${OUTPUT_DIR}/environment.txt" 2>&1

configure_and_build() {
  local name="$1"
  local build_dir="$2"
  local nvtx="$3"
  local configure_log="${OUTPUT_DIR}/${name}_configure.log"
  local build_log="${OUTPUT_DIR}/${name}_build.log"
  cmake -S "${REPO_ROOT}/app" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_BUILD_TYPE=Release -DPPE_ENABLE_NVTX="${nvtx}" \
    2>&1 | tee "${configure_log}"
  local configure_rc=${PIPESTATUS[0]}
  if [[ ${configure_rc} -ne 0 ]]; then
    printf 'result=FAIL stage=%s_configure return_code=%s\n' \
      "${name}" "${configure_rc}" > "${OUTPUT_DIR}/failure_summary.txt"
    printf '%s\n' "${configure_rc}" > "${OUTPUT_DIR}/return_code.txt"
    return "${configure_rc}"
  fi
  cmake --build "${build_dir}" \
    --target exp11_video_infer exp14_async_pipeline --parallel 2 \
    2>&1 | tee "${build_log}"
  local build_rc=${PIPESTATUS[0]}
  if [[ ${build_rc} -ne 0 ]]; then
    printf 'result=FAIL stage=%s_build return_code=%s\n' \
      "${name}" "${build_rc}" > "${OUTPUT_DIR}/failure_summary.txt"
    printf '%s\n' "${build_rc}" > "${OUTPUT_DIR}/return_code.txt"
    return "${build_rc}"
  fi
}

configure_and_build plain "${PLAIN_BUILD}" OFF || exit $?
configure_and_build nvtx "${NVTX_BUILD}" ON || exit $?

for binary in \
  "${PLAIN_BUILD}/exp11_video_infer" \
  "${PLAIN_BUILD}/exp14_async_pipeline" \
  "${NVTX_BUILD}/exp13_profiled_video_infer" \
  "${NVTX_BUILD}/exp14_async_pipeline"; do
  if [[ ! -x "${binary}" ]]; then
    printf 'result=FAIL missing_binary=%s\n' "${binary}" \
      > "${OUTPUT_DIR}/failure_summary.txt"
    printf '1\n' > "${OUTPUT_DIR}/return_code.txt"
    exit 1
  fi
done

sha256sum \
  "${PLAIN_BUILD}/exp11_video_infer" \
  "${PLAIN_BUILD}/exp14_async_pipeline" \
  "${NVTX_BUILD}/exp13_profiled_video_infer" \
  "${NVTX_BUILD}/exp14_async_pipeline" > "${OUTPUT_DIR}/binary_sha256.txt"
printf '0\n' > "${OUTPUT_DIR}/return_code.txt"
printf 'result=PASS output_dir=%s\n' "${OUTPUT_DIR}" \
  | tee "${OUTPUT_DIR}/summary.txt"
