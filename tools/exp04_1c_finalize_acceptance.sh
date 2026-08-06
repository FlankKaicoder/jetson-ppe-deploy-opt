#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="/root/autodl-tmp/jetson-ppe-deploy-opt"
VENV_DIR="$REPO_DIR/.venv-autodl"

cd "$REPO_DIR"

LATEST_STRICT_DIR=$(
    find results/model_design \
        -maxdepth 1 \
        -type d \
        -name 'exp04_1b_strict_numerical_replay_*' \
        -printf '%T@ %p\n' \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
)

if [ -z "$LATEST_STRICT_DIR" ]; then
    echo "ERROR: Exp04.1b result directory not found"
    exit 1
fi

CUDA_JSON="$LATEST_STRICT_DIR/cuda_fp32_tf32_off/summary.json"
CPU_JSON="$LATEST_STRICT_DIR/cpu_fp32/summary.json"

if [ ! -f "$CUDA_JSON" ]; then
    echo "ERROR: CUDA summary not found: $CUDA_JSON"
    exit 1
fi

if [ ! -f "$CPU_JSON" ]; then
    echo "ERROR: CPU summary not found: $CPU_JSON"
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

OUTPUT_DIR="$REPO_DIR/results/model_design/exp04_1c_reparam_acceptance_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

if [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON="$VENV_DIR/bin/python"
else
    PYTHON="$(command -v python3)"
fi

"$PYTHON" -u - \
    "$CUDA_JSON" \
    "$CPU_JSON" \
    "$OUTPUT_DIR" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


cuda_path = Path(sys.argv[1]).resolve()
cpu_path = Path(sys.argv[2]).resolve()
output_dir = Path(sys.argv[3]).resolve()

cuda = json.loads(
    cuda_path.read_text(encoding="utf-8")
)

cpu = json.loads(
    cpu_path.read_text(encoding="utf-8")
)


THRESHOLDS = {
    "initial_max_abs_error": 1e-5,
    "deploy_max_abs_error": 5e-4,
    "deploy_mean_abs_error": 1e-6,
    "deploy_relative_l2_error": 1e-6,
}


def comparison_checks(
    name: str,
    data: dict,
) -> dict[str, bool]:
    initial = data["baseline_vs_training"]
    deploy = data["training_vs_deploy"]
    baseline_deploy = data["baseline_vs_deploy"]

    return {
        f"{name}_target_indices": (
            data["target_indices"] == [17, 20]
        ),
        f"{name}_rep_training_block_count": (
            data["rep_training_block_count"] == 2
        ),
        f"{name}_converted_block_count": (
            data["converted_block_count"] == 2
        ),
        f"{name}_all_blocks_deployed": bool(
            data["all_blocks_deployed"]
        ),
        f"{name}_initial_max_abs_error": (
            initial["max_abs_error"]
            <= THRESHOLDS["initial_max_abs_error"]
        ),
        f"{name}_deploy_max_abs_error": (
            deploy["max_abs_error"]
            <= THRESHOLDS["deploy_max_abs_error"]
        ),
        f"{name}_deploy_mean_abs_error": (
            deploy["mean_abs_error"]
            <= THRESHOLDS["deploy_mean_abs_error"]
        ),
        f"{name}_deploy_relative_l2_error": (
            deploy["relative_l2_error"]
            <= THRESHOLDS[
                "deploy_relative_l2_error"
            ]
        ),
        f"{name}_baseline_deploy_max_abs_error": (
            baseline_deploy["max_abs_error"]
            <= THRESHOLDS["deploy_max_abs_error"]
        ),
        f"{name}_baseline_deploy_relative_l2": (
            baseline_deploy["relative_l2_error"]
            <= THRESHOLDS[
                "deploy_relative_l2_error"
            ]
        ),
    }


checks = {}

checks.update(
    comparison_checks(
        "cuda_fp32_tf32_off",
        cuda,
    )
)

checks.update(
    comparison_checks(
        "cpu_fp32",
        cpu,
    )
)

result = (
    "PASS"
    if all(checks.values())
    else "FAIL"
)

payload = {
    "experiment": (
        "Exp04.1c Reparameterization "
        "Integration Final Acceptance"
    ),
    "result": result,
    "cuda_source": str(cuda_path),
    "cpu_source": str(cpu_path),
    "thresholds": THRESHOLDS,
    "checks": checks,
    "cuda_metrics": {
        "initial_max_abs_error": (
            cuda["baseline_vs_training"][
                "max_abs_error"
            ]
        ),
        "deploy_max_abs_error": (
            cuda["training_vs_deploy"][
                "max_abs_error"
            ]
        ),
        "deploy_mean_abs_error": (
            cuda["training_vs_deploy"][
                "mean_abs_error"
            ]
        ),
        "deploy_relative_l2_error": (
            cuda["training_vs_deploy"][
                "relative_l2_error"
            ]
        ),
    },
    "cpu_metrics": {
        "initial_max_abs_error": (
            cpu["baseline_vs_training"][
                "max_abs_error"
            ]
        ),
        "deploy_max_abs_error": (
            cpu["training_vs_deploy"][
                "max_abs_error"
            ]
        ),
        "deploy_mean_abs_error": (
            cpu["training_vs_deploy"][
                "mean_abs_error"
            ]
        ),
        "deploy_relative_l2_error": (
            cpu["training_vs_deploy"][
                "relative_l2_error"
            ]
        ),
    },
}

(output_dir / "summary.json").write_text(
    json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)

lines = [
    "=" * 68,
    " Exp04.1c Reparameterization Integration Final Acceptance",
    "=" * 68,
    f"result={result}",
    f"cuda_source={cuda_path}",
    f"cpu_source={cpu_path}",
    "",
    "========== thresholds ==========",
    (
        "initial_max_abs_error_threshold="
        f"{THRESHOLDS['initial_max_abs_error']}"
    ),
    (
        "deploy_max_abs_error_threshold="
        f"{THRESHOLDS['deploy_max_abs_error']}"
    ),
    (
        "deploy_mean_abs_error_threshold="
        f"{THRESHOLDS['deploy_mean_abs_error']}"
    ),
    (
        "deploy_relative_l2_error_threshold="
        f"{THRESHOLDS['deploy_relative_l2_error']}"
    ),
    "",
    "========== CUDA FP32 TF32 OFF ==========",
    (
        "initial_max_abs_error="
        f"{payload['cuda_metrics']['initial_max_abs_error']:.12g}"
    ),
    (
        "deploy_max_abs_error="
        f"{payload['cuda_metrics']['deploy_max_abs_error']:.12g}"
    ),
    (
        "deploy_mean_abs_error="
        f"{payload['cuda_metrics']['deploy_mean_abs_error']:.12g}"
    ),
    (
        "deploy_relative_l2_error="
        f"{payload['cuda_metrics']['deploy_relative_l2_error']:.12g}"
    ),
    "",
    "========== CPU FP32 ==========",
    (
        "initial_max_abs_error="
        f"{payload['cpu_metrics']['initial_max_abs_error']:.12g}"
    ),
    (
        "deploy_max_abs_error="
        f"{payload['cpu_metrics']['deploy_max_abs_error']:.12g}"
    ),
    (
        "deploy_mean_abs_error="
        f"{payload['cpu_metrics']['deploy_mean_abs_error']:.12g}"
    ),
    (
        "deploy_relative_l2_error="
        f"{payload['cpu_metrics']['deploy_relative_l2_error']:.12g}"
    ),
    "",
    "========== checks ==========",
]

for name, passed in checks.items():
    lines.append(
        f"{name}={'PASS' if passed else 'FAIL'}"
    )

lines.extend(
    [
        "",
        f"overall={result}",
    ]
)

summary = "\n".join(lines) + "\n"

(output_dir / "summary.txt").write_text(
    summary,
    encoding="utf-8",
)

print(summary)

raise SystemExit(
    0 if result == "PASS" else 1
)
PY

PYTHON_RETURN_CODE=$?

echo
echo "============================================================"
echo " Exp04.1c Runner Summary"
echo "============================================================"
echo "result=$(
    grep '^overall=' "$OUTPUT_DIR/summary.txt" \
    | cut -d= -f2
)"
echo "python_return_code=$PYTHON_RETURN_CODE"
echo "output_dir=$OUTPUT_DIR"
echo "summary=$OUTPUT_DIR/summary.txt"
echo "summary_json=$OUTPUT_DIR/summary.json"
echo
echo "exp04_1c_command_completed=YES"
