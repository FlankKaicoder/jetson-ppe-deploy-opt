#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/nvidia/projects/jetson-ppe-deploy-opt

RUN_ID="exp01_2_compile_smoke_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/environment_audit/${RUN_ID}"
BUILD_DIR="${OUT_DIR}/build"

mkdir -p "${OUT_DIR}"

{
    echo "experiment: exp01_2_compile_smoke"
    echo "run_id: ${RUN_ID}"
    echo "git_branch: $(git branch --show-current)"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "build_type: Release"
    echo "cuda_architecture: 87"
} > "${OUT_DIR}/config.yaml"

set +e

cmake \
    -S experiments/exp01_compile_smoke \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    2>&1 |
    tee "${OUT_DIR}/cmake_configure.log"

CMAKE_RC=${PIPESTATUS[0]}

BUILD_RC=99
RUN_RC=99

if [ "${CMAKE_RC}" -eq 0 ]; then
    cmake \
        --build "${BUILD_DIR}" \
        --parallel "$(nproc)" \
        2>&1 |
        tee "${OUT_DIR}/build.log"

    BUILD_RC=${PIPESTATUS[0]}
else
    echo "Build skipped because CMake configuration failed." \
        > "${OUT_DIR}/build.log"
fi

if [ "${BUILD_RC}" -eq 0 ]; then
    "${BUILD_DIR}/exp01_compile_smoke" \
        2>&1 |
        tee "${OUT_DIR}/run.log"

    RUN_RC=${PIPESTATUS[0]}
else
    echo "Run skipped because build failed." \
        > "${OUT_DIR}/run.log"
fi

set -e

RESULT="FAIL"

if [ "${CMAKE_RC}" -eq 0 ] &&
   [ "${BUILD_RC}" -eq 0 ] &&
   [ "${RUN_RC}" -eq 0 ] &&
   grep -qx 'overall=PASS' "${OUT_DIR}/run.log"
then
    RESULT="PASS"
fi

{
    grep -nEi \
        '(^|[^[:alnum:]_])(error:|fatal error:|failed|undefined reference|not found)' \
        "${OUT_DIR}/cmake_configure.log" \
        "${OUT_DIR}/build.log" \
        "${OUT_DIR}/run.log" \
        || true
} > "${OUT_DIR}/abnormal.txt"

if [ ! -s "${OUT_DIR}/abnormal.txt" ]; then
    echo "No abnormal messages detected." \
        > "${OUT_DIR}/abnormal.txt"
fi

{
    echo "========== Exp01.2 Compile Smoke Test =========="
    echo "result                  : ${RESULT}"
    echo "run_id                  : ${RUN_ID}"
    echo "output_dir              : ${OUT_DIR}"
    echo "cmake_return_code       : ${CMAKE_RC}"
    echo "build_return_code       : ${BUILD_RC}"
    echo "run_return_code         : ${RUN_RC}"
    echo "git_branch              : $(git branch --show-current)"
    echo "git_commit              : $(git rev-parse HEAD)"
    echo
    echo "[RESULT] EXP01_2_COMPILE_SMOKE=${RESULT}"
} > "${OUT_DIR}/summary.txt"

echo
cat "${OUT_DIR}/summary.txt"

echo
echo "========== program output =========="
cat "${OUT_DIR}/run.log"

echo
echo "========== abnormal =========="
cat "${OUT_DIR}/abnormal.txt"

echo
echo "========== shell safety =========="
echo "EXP01_2_SCRIPT_FINISHED"

exit "${RUN_RC}"
