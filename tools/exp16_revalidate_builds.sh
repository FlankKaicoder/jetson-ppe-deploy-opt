#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
report_dir="${PPE_OUTPUT_DIR:-${repo_root}/results/plugin/exp16_revalidate_build_matrix_${timestamp}}"
onnx="/home/nvidia/models/jetson-ppe/exp06/yolo11n_baseline_exp06_b1_640_opset17.onnx"
plugin_onnx="${repo_root}/results/plugin/exp16_1_graph_smoke_20260809_163112/yolo11n_exp16_plugin.onnx"
plugin_so="${repo_root}/results/plugin/exp16_8_rebuild_20260809_170133/build/libppe_yolo_decode_plugin.so"
plugin_builder="${repo_root}/results/plugin/exp16_8_rebuild_20260809_170133/build/exp16_build_engine"
trtexec="/usr/src/tensorrt/bin/trtexec"

expected_onnx="305e23c65aa3b1d01e7b1a784c355665228f435f0b904b92ed2618954736d1f8"
expected_plugin_onnx="cfea7b13bb11cb11f8026f0f327d8939167a6b991cdb6e2ec8b038a014736d04"
expected_plugin_so="b5d402ffab879758c16289c7c385ddab7eeaa0c270a76f65042eb35a5e8477f1"

fail_early() {
    printf 'result=FAIL\nreason=%s\n' "$1" > "${report_dir}/failure_summary.txt"
    printf '1\n' > "${report_dir}/return_code.txt"
    exit 1
}

if [[ -e "${report_dir}" ]]; then
    echo "ERROR: output already exists: ${report_dir}" >&2
    exit 1
fi
mkdir -p "${report_dir}"
[[ -x "${trtexec}" ]] || fail_early "trtexec missing"
[[ -x "${plugin_builder}" ]] || fail_early "Plugin builder missing"
[[ "$(sha256sum "${onnx}" | awk '{print $1}')" == "${expected_onnx}" ]] || \
    fail_early "frozen ONNX hash mismatch"
[[ "$(sha256sum "${plugin_onnx}" | awk '{print $1}')" == "${expected_plugin_onnx}" ]] || \
    fail_early "Plugin ONNX hash mismatch"
[[ "$(sha256sum "${plugin_so}" | awk '{print $1}')" == "${expected_plugin_so}" ]] || \
    fail_early "Plugin shared library hash mismatch"

{
    hostname
    uname -m
    git rev-parse HEAD
    git branch --show-current
    dpkg-query -W libnvinfer10 nvinfer-bin
    /usr/local/cuda/bin/nvcc --version
    nvpmodel -q
    sha256sum "${onnx}" "${plugin_onnx}" "${plugin_so}"
    echo "build_order=B1_then_P_then_B2"
    echo "workspace_mib=1024"
    echo "builder_optimization_level=3"
    echo "tf32=disabled"
} > "${report_dir}/environment.txt" 2>&1

build_baseline() {
    local name="$1"
    local engine="${report_dir}/${name}.engine"
    local start end rc
    start="$(date +%s)"
    set +e
    "${trtexec}" \
        --onnx="${onnx}" \
        --saveEngine="${engine}" \
        --fp16 \
        --noTF32 \
        --memPoolSize=workspace:1024 \
        --builderOptimizationLevel=3 \
        --profilingVerbosity=detailed \
        --skipInference \
        > "${report_dir}/${name}_build.log" 2>&1
    rc=$?
    set -e
    end="$(date +%s)"
    printf '%s\n' "${rc}" > "${report_dir}/${name}_return_code.txt"
    printf '%s\n' "$((end - start))" > "${report_dir}/${name}_build_seconds.txt"
    [[ "${rc}" -eq 0 && -s "${engine}" ]] || fail_early "${name} build failed"
}

build_plugin() {
    local engine="${report_dir}/P.engine"
    local start end rc
    start="$(date +%s)"
    set +e
    "${plugin_builder}" "${plugin_so}" "${plugin_onnx}" "${engine}" \
        > "${report_dir}/P_build.log" 2>&1
    rc=$?
    set -e
    end="$(date +%s)"
    printf '%s\n' "${rc}" > "${report_dir}/P_return_code.txt"
    printf '%s\n' "$((end - start))" > "${report_dir}/P_build_seconds.txt"
    [[ "${rc}" -eq 0 && -s "${engine}" ]] || fail_early "P build failed"
}

set -e
build_baseline B1
build_plugin
build_baseline B2
sha256sum "${report_dir}/B1.engine" "${report_dir}/P.engine" \
    "${report_dir}/B2.engine" > "${report_dir}/sha256.txt"
printf 'result=PASS\noutput_dir=%s\n' "${report_dir}" > "${report_dir}/summary.txt"
printf '0\n' > "${report_dir}/return_code.txt"
echo "exp16_revalidate_build_matrix=PASS output=${report_dir}"
