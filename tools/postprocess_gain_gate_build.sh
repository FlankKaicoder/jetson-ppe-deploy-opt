#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${repo_root}/results/gpu_postprocess/postprocess_gain_gate_build_${timestamp}"
build_dir="${repo_root}/build/postprocess_gain_gate"
mkdir -p "${output_dir}"

{
  hostname
  whoami
  uname -a
  /usr/local/cuda/bin/nvcc --version
  dpkg-query -W nvinfer-bin libnvinfer10 2>/dev/null
  cmake --version | head -1
  g++ --version | head -1
  git -C "${repo_root}" rev-parse HEAD
  git -C "${repo_root}" branch --show-current
  git -C "${repo_root}" status --short --untracked-files=no
} > "${output_dir}/environment.txt" 2>&1

cmake -S "${repo_root}/app" -B "${build_dir}" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DPPE_ENABLE_NVTX=OFF \
  > "${output_dir}/configure.log" 2>&1
configure_rc=$?
printf '%s\n' "${configure_rc}" > "${output_dir}/configure_return_code.txt"
if [[ ${configure_rc} -ne 0 ]]; then
  printf 'result=FAIL stage=configure output_dir=%s\n' "${output_dir}" \
    > "${output_dir}/failure_summary.txt"
  exit "${configure_rc}"
fi

cmake --build "${build_dir}" \
  --target exp11_video_infer exp15_gpu_postprocess exp15_gpu_postprocess_test \
    postprocess_gain_gate_benchmark --parallel 2 \
  > "${output_dir}/build.log" 2>&1
build_rc=$?
printf '%s\n' "${build_rc}" > "${output_dir}/build_return_code.txt"
if [[ ${build_rc} -ne 0 ]]; then
  printf 'result=FAIL stage=build output_dir=%s\n' "${output_dir}" \
    > "${output_dir}/failure_summary.txt"
  exit "${build_rc}"
fi

"${build_dir}/cuda/exp15_gpu_postprocess_test" \
  --output-dir "${output_dir}/synthetic" \
  > "${output_dir}/synthetic.log" 2>&1
synthetic_rc=$?
printf '%s\n' "${synthetic_rc}" > "${output_dir}/synthetic_return_code.txt"
if [[ ${synthetic_rc} -ne 0 ]]; then
  printf 'result=FAIL stage=synthetic output_dir=%s\n' "${output_dir}" \
    > "${output_dir}/failure_summary.txt"
  exit "${synthetic_rc}"
fi

sha256sum \
  "${build_dir}/exp15_gpu_postprocess" \
  "${build_dir}/cuda/exp15_gpu_postprocess_test" \
  "${build_dir}/cuda/postprocess_gain_gate_benchmark" \
  "${output_dir}/synthetic/summary.json" \
  "${output_dir}/synthetic/synthetic_results.csv" \
  > "${output_dir}/sha256.txt"
printf 'result=PASS output_dir=%s\n' "${output_dir}" \
  | tee "${output_dir}/summary.txt"
