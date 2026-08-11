#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${repo_root}/results/final_benchmark/exp19_0_build_${timestamp}"
build_dir="${repo_root}/build/exp19_plain"
mkdir -p "${output_dir}"

{
  hostname
  whoami
  uname -a
  git -C "${repo_root}" branch --show-current
  git -C "${repo_root}" rev-parse HEAD
  /usr/local/cuda/bin/nvcc --version | tail -1
  dpkg-query -W libnvinfer10 2>/dev/null || true
  cmake --version | head -1
  g++ --version | head -1
} > "${output_dir}/environment.txt" 2>&1

cmake -S "${repo_root}/app" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DPPE_ENABLE_NVTX=OFF \
  > "${output_dir}/configure.log" 2>&1
configure_rc=$?
printf '%s\n' "${configure_rc}" > "${output_dir}/configure_return_code.txt"
if [[ ${configure_rc} -ne 0 ]]; then
  printf 'result=FAIL stage=configure return_code=%s\n' "${configure_rc}" \
    > "${output_dir}/failure_summary.txt"
  exit "${configure_rc}"
fi

cmake --build "${build_dir}" --target exp15_gpu_postprocess -j2 \
  > "${output_dir}/build.log" 2>&1
build_rc=$?
printf '%s\n' "${build_rc}" > "${output_dir}/build_return_code.txt"
binary="${build_dir}/exp15_gpu_postprocess"
if [[ ${build_rc} -ne 0 || ! -x "${binary}" ]]; then
  printf 'result=FAIL stage=build return_code=%s\n' "${build_rc}" \
    > "${output_dir}/failure_summary.txt"
  exit 1
fi
sha256sum "${binary}" > "${output_dir}/binary_sha256.txt"
printf 'result=PASS output_dir=%s binary=%s\n' "${output_dir}" "${binary}" \
  | tee "${output_dir}/summary.txt"
