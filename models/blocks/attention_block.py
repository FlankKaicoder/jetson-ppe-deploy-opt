from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules import ChannelAttention, SpatialAttention
except ImportError:
    from ultralytics.nn.modules.conv import ChannelAttention, SpatialAttention


class ResidualCBAMLite(nn.Module):
    """A deployment-friendly CBAM branch with identity initialization."""

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, received {channels}")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.channel_attention = ChannelAttention(self.channels)
        self.spatial_attention = SpatialAttention(self.kernel_size)
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.spatial_attention(self.channel_attention(x))
        return x + self.residual_scale * attended


class P3AttentionWrapper(nn.Module):
    """Wrap the existing P3 Neck block without changing graph indices."""

    def __init__(self, source: nn.Module, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.source = source
        self.channels = int(channels)
        self.attention = ResidualCBAMLite(self.channels, kernel_size=kernel_size)

        reference_parameter = next(source.parameters())
        self.attention.to(device=reference_parameter.device, dtype=reference_parameter.dtype)

        for attribute in ("i", "f"):
            if hasattr(source, attribute):
                setattr(self, attribute, copy.deepcopy(getattr(source, attribute)))

        self.type = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self.np = sum(parameter.numel() for parameter in self.parameters())
        self.train(source.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(self.source(x))


def _detect_p3_index(detection_model: nn.Module) -> int:
    if not hasattr(detection_model, "model"):
        raise TypeError("detection model does not contain a top-level .model graph")
    detect = detection_model.model[-1]
    sources = getattr(detect, "f", None)
    if not isinstance(sources, (list, tuple)) or len(sources) != 3:
        raise RuntimeError(f"unexpected Detect source indices: {sources}")
    p3_index = int(sources[0])
    if p3_index != 16:
        raise RuntimeError(f"expected YOLO11 P3 Neck index 16, received {p3_index}")
    return p3_index


def _output_channels(source: nn.Module) -> int:
    cv2 = getattr(source, "cv2", None)
    conv = getattr(cv2, "conv", None)
    channels = getattr(conv, "out_channels", None)
    if channels is None:
        raise RuntimeError(
            "unable to infer P3 output channels from the expected YOLO11 C3k2 block"
        )
    return int(channels)


def install_p3_attention(
    detection_model: nn.Module,
    kernel_size: int = 7,
) -> dict[str, Any]:
    """Install one identity-initialized attention wrapper after pretrained loading."""

    p3_index = _detect_p3_index(detection_model)
    source = detection_model.model[p3_index]
    if isinstance(source, P3AttentionWrapper):
        return describe_p3_attention(detection_model)[0]

    channels = _output_channels(source)
    source_parameters = sum(parameter.numel() for parameter in source.parameters())
    wrapper = P3AttentionWrapper(source, channels=channels, kernel_size=kernel_size)
    detection_model.model[p3_index] = wrapper

    return {
        "index": p3_index,
        "source_class": source.__class__.__name__,
        "replacement_class": wrapper.__class__.__name__,
        "channels": channels,
        "kernel_size": int(kernel_size),
        "source_parameter_count": source_parameters,
        "wrapper_parameter_count": sum(parameter.numel() for parameter in wrapper.parameters()),
        "added_parameter_count": sum(parameter.numel() for parameter in wrapper.attention.parameters()),
        "residual_scale_initial": float(wrapper.attention.residual_scale.detach().cpu()),
    }


def count_p3_attention_wrappers(module: nn.Module) -> int:
    return sum(1 for item in module.modules() if isinstance(item, P3AttentionWrapper))


def describe_p3_attention(detection_model: nn.Module) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    if not hasattr(detection_model, "model"):
        return details
    for index, module in enumerate(detection_model.model):
        if not isinstance(module, P3AttentionWrapper):
            continue
        details.append(
            {
                "index": index,
                "source_class": module.source.__class__.__name__,
                "replacement_class": module.__class__.__name__,
                "channels": module.channels,
                "kernel_size": module.attention.kernel_size,
                "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
                "attention_parameter_count": sum(
                    parameter.numel() for parameter in module.attention.parameters()
                ),
                "residual_scale": float(module.attention.residual_scale.detach().cpu()),
            }
        )
    return details
