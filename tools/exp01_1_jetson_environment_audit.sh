#!/usr/bin/env bash
set -Eeuo pipefail

OUT_DIR="${1:?usage: $0 OUT_DIR}"
mkdir -p "${OUT_DIR}/artifacts"

ENV_FILE="${OUT_DIR}/environment.txt"
SUMMARY_FILE="${OUT_DIR}/summary.txt"
ABNORMAL_FILE="${OUT_DIR}/abnormal.txt"
PACKAGES_FILE="${OUT_DIR}/artifacts/packages.txt"
GSTREAMER_FILE="${OUT_DIR}/artifacts/gstreamer_plugins.txt"
POWER_FILE="${OUT_DIR}/artifacts/power_status.txt"
CAMERA_FILE="${OUT_DIR}/artifacts/camera_status.txt"
GIT_FILE="${OUT_DIR}/artifacts/git_snapshot.txt"

: > "${ENV_FILE}"
: > "${SUMMARY_FILE}"
: > "${ABNORMAL_FILE}"
: > "${PACKAGES_FILE}"
: > "${GSTREAMER_FILE}"
: > "${POWER_FILE}"
: > "${CAMERA_FILE}"
: > "${GIT_FILE}"

section()
{
    printf '\n========== %s ==========\n' "$1" >> "${ENV_FILE}"
}

record_command()
{
    local title="$1"
    shift

    section "${title}"
    printf '$ ' >> "${ENV_FILE}"
    printf '%q ' "$@" >> "${ENV_FILE}"
    printf '\n' >> "${ENV_FILE}"

    "$@" >> "${ENV_FILE}" 2>&1 || true
}

check_command()
{
    local command_name="$1"

    if command -v "${command_name}" >/dev/null 2>&1; then
        echo "PASS"
    else
        echo "MISSING"
    fi
}

check_gst_plugin()
{
    local plugin_name="$1"

    if command -v gst-inspect-1.0 >/dev/null 2>&1 &&
       gst-inspect-1.0 "${plugin_name}" >/dev/null 2>&1; then
        echo "PASS"
    else
        echo "MISSING"
    fi
}

MODEL="$(
    tr -d '\0' < /proc/device-tree/model 2>/dev/null ||
    echo "unknown"
)"

L4T="$(
    head -n 1 /etc/nv_tegra_release 2>/dev/null ||
    echo "not-found"
)"

JETPACK="$(
    dpkg-query \
        -W \
        -f='${Version}' \
        nvidia-jetpack \
        2>/dev/null ||
    echo "metapackage-not-installed"
)"

CUDA_STATUS="$(check_command nvcc)"
PYTHON_STATUS="$(check_command python3)"
GSTREAMER_STATUS="$(check_command gst-launch-1.0)"
ARGUS_STATUS="$(check_gst_plugin nvarguscamerasrc)"
H264ENC_STATUS="$(check_gst_plugin nvv4l2h264enc)"
H265ENC_STATUS="$(check_gst_plugin nvv4l2h265enc)"

TRTEXEC_PATH=""

if command -v trtexec >/dev/null 2>&1; then
    TRTEXEC_PATH="$(command -v trtexec)"
elif [ -x /usr/src/tensorrt/bin/trtexec ]; then
    TRTEXEC_PATH="/usr/src/tensorrt/bin/trtexec"
fi

if [ -n "${TRTEXEC_PATH}" ]; then
    TRT_STATUS="PASS"
else
    TRT_STATUS="MISSING"
fi

GIT_TOP="$(git rev-parse --show-toplevel 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_TAGS="$(git tag --points-at HEAD 2>/dev/null | tr '\n' ' ')"

record_command "date" date --iso-8601=seconds
record_command "hostname" hostname
record_command "architecture" uname -m
record_command "kernel" uname -a
record_command "operating system" cat /etc/os-release
record_command "Jetson model" bash -c \
    "tr -d '\\0' < /proc/device-tree/model 2>/dev/null || true"
record_command "L4T release" bash -c \
    "cat /etc/nv_tegra_release 2>/dev/null || true"
record_command "CPU information" lscpu
record_command "memory" free -h
record_command "disk" df -h /
record_command "NVMe devices" lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS

record_command "CUDA compiler" bash -c \
    "command -v nvcc && nvcc --version || true"
record_command "CUDA runtime library" bash -c \
    "ldconfig -p 2>/dev/null | grep -E 'libcudart\\.so' || true"

{
    echo "========== NVIDIA packages =========="
    dpkg-query -W \
        -f='${binary:Package}\t${Version}\n' \
        'nvidia-jetpack' \
        'cuda-*' \
        'libcudnn*' \
        'libnvinfer*' \
        'python3-libnvinfer*' \
        2>/dev/null |
        sort -u || true
} > "${PACKAGES_FILE}"

record_command "selected NVIDIA packages" cat "${PACKAGES_FILE}"

if [ -n "${TRTEXEC_PATH}" ]; then
    record_command "TensorRT trtexec" \
        "${TRTEXEC_PATH}" --version
else
    section "TensorRT trtexec"
    echo "trtexec not found" >> "${ENV_FILE}"
fi

record_command "Python version" python3 --version

section "Python package versions"
python3 - <<'PY' >> "${ENV_FILE}" 2>&1 || true
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

packages = [
    ("numpy", "numpy"),
    ("opencv-python", "cv2"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("ultralytics", "ultralytics"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("tensorrt", "tensorrt"),
    ("pycuda", "pycuda"),
]

for distribution, module_name in packages:
    package_version = None

    try:
        package_version = version(distribution)
    except PackageNotFoundError:
        pass

    try:
        module = import_module(module_name)
        module_version = getattr(module, "__version__", package_version)
        print(f"{module_name:16s}: {module_version or 'installed'}")
    except Exception as exc:
        print(
            f"{module_name:16s}: NOT_IMPORTABLE "
            f"({type(exc).__name__}: {exc})"
        )
PY

{
    echo "========== GStreamer version =========="
    gst-launch-1.0 --version 2>&1 || true

    echo
    echo "========== nvarguscamerasrc =========="
    gst-inspect-1.0 nvarguscamerasrc 2>&1 |
        grep -E \
        'Factory Details|Long-name|Description|Filename|Version' ||
        true

    echo
    echo "========== nvv4l2h264enc =========="
    gst-inspect-1.0 nvv4l2h264enc 2>&1 |
        grep -E \
        'Factory Details|Long-name|Description|Filename|Version' ||
        true

    echo
    echo "========== nvv4l2h265enc =========="
    gst-inspect-1.0 nvv4l2h265enc 2>&1 |
        grep -E \
        'Factory Details|Long-name|Description|Filename|Version' ||
        true
} > "${GSTREAMER_FILE}"

record_command "GStreamer core and NVIDIA plugins" \
    cat "${GSTREAMER_FILE}"

{
    echo "========== nvpmodel =========="
    nvpmodel -q 2>&1 || true

    echo
    echo "========== jetson_clocks =========="
    jetson_clocks --show 2>&1 || true

    echo
    echo "========== tegrastats sample =========="
    if command -v tegrastats >/dev/null 2>&1; then
        timeout 3s tegrastats --interval 1000 2>&1 || true
    else
        echo "tegrastats not found"
    fi
} > "${POWER_FILE}"

record_command "power and clock status" cat "${POWER_FILE}"

{
    echo "========== video devices =========="
    ls -l /dev/video* 2>&1 || true

    echo
    echo "========== media devices =========="
    ls -l /dev/media* 2>&1 || true

    echo
    echo "========== v4l2 devices =========="
    if command -v v4l2-ctl >/dev/null 2>&1; then
        v4l2-ctl --list-devices 2>&1 || true
    else
        echo "v4l2-ctl not found"
    fi

    echo
    echo "========== Argus daemon =========="
    systemctl is-active nvargus-daemon 2>&1 || true
} > "${CAMERA_FILE}"

record_command "camera status" cat "${CAMERA_FILE}"

{
    echo "repository=${GIT_TOP}"
    echo "branch=${GIT_BRANCH}"
    echo "commit=${GIT_COMMIT}"
    echo "tags=${GIT_TAGS}"
    echo
    git status --short
    echo
    git log --oneline --decorate -n 5
} > "${GIT_FILE}"

record_command "Git snapshot" cat "${GIT_FILE}"

OVERALL="PASS"

for item in \
    "CUDA nvcc:${CUDA_STATUS}" \
    "TensorRT trtexec:${TRT_STATUS}" \
    "Python 3:${PYTHON_STATUS}" \
    "GStreamer:${GSTREAMER_STATUS}" \
    "nvarguscamerasrc:${ARGUS_STATUS}" \
    "nvv4l2h264enc:${H264ENC_STATUS}"
do
    name="${item%%:*}"
    status="${item##*:}"

    if [ "${status}" != "PASS" ]; then
        OVERALL="PARTIAL"
        echo "${name}: ${status}" >> "${ABNORMAL_FILE}"
    fi
done

if [ "${H265ENC_STATUS}" != "PASS" ]; then
    echo "nvv4l2h265enc: ${H265ENC_STATUS}" >> "${ABNORMAL_FILE}"
fi

if [ ! -s "${ABNORMAL_FILE}" ]; then
    echo "No missing core components detected." > "${ABNORMAL_FILE}"
fi

{
    echo "========== Exp01.1 Jetson Environment Audit =========="
    echo "result                  : ${OVERALL}"
    echo "timestamp               : $(date --iso-8601=seconds)"
    echo "repository              : ${GIT_TOP}"
    echo "branch                  : ${GIT_BRANCH}"
    echo "commit                  : ${GIT_COMMIT}"
    echo "tags                    : ${GIT_TAGS}"
    echo "jetson_model            : ${MODEL}"
    echo "l4t                     : ${L4T}"
    echo "jetpack_metapackage     : ${JETPACK}"
    echo "cuda_nvcc               : ${CUDA_STATUS}"
    echo "tensorrt_trtexec        : ${TRT_STATUS}"
    echo "trtexec_path            : ${TRTEXEC_PATH:-not-found}"
    echo "python3                 : ${PYTHON_STATUS}"
    echo "gstreamer               : ${GSTREAMER_STATUS}"
    echo "nvarguscamerasrc        : ${ARGUS_STATUS}"
    echo "nvv4l2h264enc           : ${H264ENC_STATUS}"
    echo "nvv4l2h265enc           : ${H265ENC_STATUS}"
    echo "environment_file        : ${ENV_FILE}"
    echo "abnormal_file           : ${ABNORMAL_FILE}"
    echo
    echo "[RESULT] EXP01_1_JETSON_AUDIT=${OVERALL}"
} > "${SUMMARY_FILE}"

echo "stage,result" > "${OUT_DIR}/metrics.csv"
echo "jetson_environment_audit,${OVERALL}" >> "${OUT_DIR}/metrics.csv"

cat "${SUMMARY_FILE}"

echo
echo "========== abnormal =========="
cat "${ABNORMAL_FILE}"
