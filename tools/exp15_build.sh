#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${repo_root}/results/gpu_postprocess/exp15_0_build_${timestamp}"
mkdir -p "${output_dir}"

build_one() {
  local name="$1"
  local nvtx="$2"
  local build_dir="${repo_root}/build/exp15_${name}"
  cmake -S "${repo_root}/app" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DPPE_ENABLE_NVTX="${nvtx}" \
    > "${output_dir}/configure_${name}.log" 2>&1
  local rc=$?
  printf '%s\n' "${rc}" > "${output_dir}/configure_${name}_return_code.txt"
  if [[ ${rc} -ne 0 ]]; then
    printf 'result=FAIL stage=configure build=%s\n' "${name}" \
      > "${output_dir}/failure_summary.txt"
    return "${rc}"
  fi
  cmake --build "${build_dir}" \
    --target exp11_video_infer exp15_gpu_postprocess exp15_gpu_postprocess_test \
    --parallel 2 > "${output_dir}/build_${name}.log" 2>&1
  rc=$?
  printf '%s\n' "${rc}" > "${output_dir}/build_${name}_return_code.txt"
  if [[ ${rc} -ne 0 ]]; then
    printf 'result=FAIL stage=build build=%s\n' "${name}" \
      > "${output_dir}/failure_summary.txt"
    return "${rc}"
  fi
}

build_one plain OFF || exit $?
build_one nvtx ON || exit $?

"${repo_root}/build/exp15_plain/cuda/exp15_gpu_postprocess_test" \
  --output-dir "${output_dir}/synthetic" \
  > "${output_dir}/synthetic.log" 2>&1
test_rc=$?
printf '%s\n' "${test_rc}" > "${output_dir}/synthetic_return_code.txt"
if [[ ${test_rc} -ne 0 ]]; then
  printf 'result=FAIL stage=synthetic\n' > "${output_dir}/failure_summary.txt"
  exit "${test_rc}"
fi

sha256sum \
  "${repo_root}/build/exp15_plain/exp11_video_infer" \
  "${repo_root}/build/exp15_plain/exp15_gpu_postprocess" \
  "${repo_root}/build/exp15_plain/cuda/exp15_gpu_postprocess_test" \
  "${output_dir}/synthetic/summary.json" \
  "${output_dir}/synthetic/synthetic_results.csv" \
  > "${output_dir}/sha256.txt"
printf 'result=PASS output_dir=%s\n' "${output_dir}" \
  | tee "${output_dir}/summary.txt"
