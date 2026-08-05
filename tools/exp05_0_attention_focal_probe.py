from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from models.blocks.attention_block import (
    P3AttentionWrapper,
    count_p3_attention_wrappers,
    describe_p3_attention,
    install_p3_attention,
)
from models.exp05_trainer import (
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    ElementwiseFocalBCE,
    FocalDetectionLoss,
)


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        tensors: list[torch.Tensor] = []
        for key in sorted(value):
            tensors.extend(flatten_tensors(value[key]))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(flatten_tensors(item))
        return tensors
    return []


def compare_outputs(reference: Any, candidate: Any) -> dict[str, float | int]:
    reference_tensors = flatten_tensors(reference)
    candidate_tensors = flatten_tensors(candidate)
    if len(reference_tensors) != len(candidate_tensors):
        raise RuntimeError("output tensor count mismatch")
    maximum = 0.0
    absolute_sum = 0.0
    element_count = 0
    for first, second in zip(reference_tensors, candidate_tensors):
        if first.shape != second.shape:
            raise RuntimeError(f"output shape mismatch: {first.shape} != {second.shape}")
        absolute = (first - second).abs()
        maximum = max(maximum, float(absolute.max()) if absolute.numel() else 0.0)
        absolute_sum += float(absolute.sum())
        element_count += absolute.numel()
    return {
        "tensor_count": len(reference_tensors),
        "element_count": element_count,
        "max_abs_error": maximum,
        "mean_abs_error": absolute_sum / max(element_count, 1),
    }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def build_nc3_reference(pretrained: Path) -> DetectionModel:
    source = YOLO(str(pretrained)).model
    model = DetectionModel(copy.deepcopy(source.yaml), ch=3, nc=3, verbose=False)
    model.load(source)
    model.args = copy.deepcopy(source.args)
    return model


def attention_probe(reference: DetectionModel) -> dict[str, Any]:
    candidate = copy.deepcopy(reference)
    manifest = install_p3_attention(candidate)
    reference.eval()
    candidate.eval()
    generator = torch.Generator().manual_seed(42)
    sample = torch.randn(1, 3, 640, 640, generator=generator)
    with torch.no_grad():
        baseline_output = reference(sample)
        attention_output = candidate(sample)
    comparison = compare_outputs(baseline_output, attention_output)

    train_candidate = copy.deepcopy(candidate).train()
    wrapper = train_candidate.model[16]
    if not isinstance(wrapper, P3AttentionWrapper):
        raise RuntimeError("layer 16 is not the expected P3AttentionWrapper")
    train_candidate.zero_grad(set_to_none=True)
    training_sample = torch.randn(2, 3, 64, 64, generator=generator)
    scalar = sum(tensor.float().mean() for tensor in flatten_tensors(train_candidate(training_sample)))
    scalar.backward()
    scale_gradient = wrapper.attention.residual_scale.grad
    if scale_gradient is None:
        raise RuntimeError("residual scale did not receive a gradient")

    wrapper.attention.residual_scale.data.fill_(0.1)
    train_candidate.zero_grad(set_to_none=True)
    scalar = sum(tensor.float().mean() for tensor in flatten_tensors(train_candidate(training_sample)))
    scalar.backward()
    attention_gradients = [
        parameter.grad
        for name, parameter in wrapper.attention.named_parameters()
        if name != "residual_scale"
    ]
    attention_gradient_abs_sum = sum(
        float(gradient.detach().abs().sum())
        for gradient in attention_gradients
        if gradient is not None
    )

    with tempfile.TemporaryDirectory(prefix="exp05_attention_probe_") as temporary:
        checkpoint = Path(temporary) / "attention_model.pt"
        torch.save(candidate, checkpoint)
        restored = torch.load(checkpoint, map_location="cpu", weights_only=False).eval()
        with torch.no_grad():
            restored_output = restored(sample)
        roundtrip = compare_outputs(attention_output, restored_output)

    result = {
        "manifest": manifest,
        "description": describe_p3_attention(candidate),
        "wrapper_count": count_p3_attention_wrappers(candidate),
        "baseline_parameters": parameter_count(reference),
        "attention_parameters": parameter_count(candidate),
        "added_parameters": parameter_count(candidate) - parameter_count(reference),
        "identity_comparison": comparison,
        "residual_scale_gradient": float(scale_gradient.detach()),
        "attention_gradient_abs_sum_at_scale_0_1": attention_gradient_abs_sum,
        "serialization_comparison": roundtrip,
    }
    result["pass"] = bool(
        result["wrapper_count"] == 1
        and result["added_parameters"] == 4259
        and comparison["max_abs_error"] == 0.0
        and abs(result["residual_scale_gradient"]) > 0.0
        and attention_gradient_abs_sum > 0.0
        and roundtrip["max_abs_error"] == 0.0
    )
    return result


def focal_probe(reference: DetectionModel) -> dict[str, Any]:
    focal = ElementwiseFocalBCE(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(2, 100, 3, generator=generator, requires_grad=True)
    targets = torch.rand(2, 100, 3, generator=generator)
    focal_values = focal(logits, targets)
    bce_values = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    focal_values.sum().backward()

    easy_logit = torch.tensor([-6.0])
    hard_logit = torch.tensor([2.0])
    negative_target = torch.zeros(1)
    easy_ratio = float(focal(easy_logit, negative_target) / torch.nn.functional.binary_cross_entropy_with_logits(easy_logit, negative_target))
    hard_ratio = float(focal(hard_logit, negative_target) / torch.nn.functional.binary_cross_entropy_with_logits(hard_logit, negative_target))

    criterion = FocalDetectionLoss(reference)
    result = {
        "gamma": focal.gamma,
        "alpha": focal.alpha,
        "input_shape": list(logits.shape),
        "focal_shape": list(focal_values.shape),
        "bce_shape": list(bce_values.shape),
        "gradient_abs_sum": float(logits.grad.detach().abs().sum()),
        "easy_negative_focal_to_bce_ratio": easy_ratio,
        "hard_negative_focal_to_bce_ratio": hard_ratio,
        "criterion_class": f"{criterion.__class__.__module__}.{criterion.__class__.__name__}",
        "bbox_loss_class": criterion.bbox_loss.__class__.__name__,
        "classification_reduction": criterion.bce.reduction,
    }
    result["pass"] = bool(
        result["focal_shape"] == result["input_shape"]
        and result["bce_shape"] == result["input_shape"]
        and result["gradient_abs_sum"] > 0.0
        and easy_ratio < hard_ratio
        and criterion.bce is not None
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()

    pretrained = Path(args.pretrained).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    reference = build_nc3_reference(pretrained)
    attention = attention_probe(reference)
    focal = focal_probe(reference)
    summary = {
        "pretrained": str(pretrained),
        "attention": attention,
        "focal": focal,
        "overall": "PASS" if attention["pass"] and focal["pass"] else "FAIL",
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "Exp05.0 Attention and Focal Probe",
        f"overall={summary['overall']}",
        f"attention_pass={attention['pass']}",
        f"focal_pass={focal['pass']}",
        f"p3_index={attention['manifest']['index']}",
        f"p3_channels={attention['manifest']['channels']}",
        f"added_parameters={attention['added_parameters']}",
        f"identity_max_abs_error={attention['identity_comparison']['max_abs_error']}",
        f"serialization_max_abs_error={attention['serialization_comparison']['max_abs_error']}",
        f"residual_scale_gradient={attention['residual_scale_gradient']}",
        f"attention_gradient_abs_sum_at_scale_0_1={attention['attention_gradient_abs_sum_at_scale_0_1']}",
        f"focal_gamma={focal['gamma']}",
        f"focal_alpha={focal['alpha']}",
        f"easy_negative_focal_to_bce_ratio={focal['easy_negative_focal_to_bce_ratio']}",
        f"hard_negative_focal_to_bce_ratio={focal['hard_negative_focal_to_bce_ratio']}",
    ]
    (report_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if summary["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
