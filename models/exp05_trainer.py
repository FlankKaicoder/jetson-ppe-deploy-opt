from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics.models.yolo.detect import DetectionTrainer
except ImportError:
    from ultralytics.models.yolo.detect.train import DetectionTrainer

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

from models.blocks.attention_block import (
    count_p3_attention_wrappers,
    describe_p3_attention,
    install_p3_attention,
)


FOCAL_GAMMA = 1.5
FOCAL_ALPHA = 0.25


class ElementwiseFocalBCE(nn.Module):
    """Sigmoid focal BCE that preserves the elementwise BCE output shape."""

    reduction = "none"

    def __init__(self, gamma: float = FOCAL_GAMMA, alpha: float = FOCAL_ALPHA) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("focal gamma must be non-negative")
        if not 0 <= alpha <= 1:
            raise ValueError("focal alpha must be in [0, 1]")
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=pred.dtype)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        probability = pred.sigmoid()
        p_t = target * probability + (1.0 - target) * (1.0 - probability)
        modulating_factor = (1.0 - p_t).pow(self.gamma)
        alpha_factor = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        return loss * modulating_factor * alpha_factor


class FocalDetectionLoss(v8DetectionLoss):
    """YOLO detection loss with only the classification BCE replaced."""

    def __init__(
        self,
        model: nn.Module,
        gamma: float = FOCAL_GAMMA,
        alpha: float = FOCAL_ALPHA,
    ) -> None:
        super().__init__(model)
        self.bce = ElementwiseFocalBCE(gamma=gamma, alpha=alpha)


class FocalDetectionModel(DetectionModel):
    """DetectionModel whose training criterion changes only cls BCE to focal BCE."""

    def init_criterion(self) -> FocalDetectionLoss:
        return FocalDetectionLoss(self)


def _build_focal_model(
    trainer: DetectionTrainer,
    cfg: Any,
    weights: Any,
    verbose: bool,
) -> FocalDetectionModel:
    model = trainer.set_model_names_for_load(
        FocalDetectionModel(
            cfg,
            nc=trainer.data["nc"],
            ch=trainer.data["channels"],
            verbose=verbose,
        )
    )
    if weights:
        model.load(weights)
    return model


class AttentionDetectionTrainer(DetectionTrainer):
    """Detection trainer that adds one P3 attention block after weight loading."""

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> nn.Module:
        if isinstance(weights, nn.Module) and count_p3_attention_wrappers(weights) == 1:
            model = copy.deepcopy(weights)
            self.attention_source = "existing_attention_model"
            self.attention_manifest = describe_p3_attention(model)
            return model

        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        self.attention_manifest = [install_p3_attention(model)]
        self.attention_source = "normal_model_after_pretrained_load"
        if count_p3_attention_wrappers(model) != 1:
            raise RuntimeError("expected exactly one P3 attention wrapper")
        return model


class FocalDetectionTrainer(DetectionTrainer):
    """Detection trainer using the focal-only detection model."""

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> nn.Module:
        if isinstance(weights, FocalDetectionModel):
            return copy.deepcopy(weights)
        return _build_focal_model(self, cfg=cfg, weights=weights, verbose=verbose)


class AttentionFocalDetectionTrainer(DetectionTrainer):
    """Combined trainer reserved for Exp05.3 after single-variable acceptance."""

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> nn.Module:
        if (
            isinstance(weights, FocalDetectionModel)
            and count_p3_attention_wrappers(weights) == 1
        ):
            model = copy.deepcopy(weights)
            self.attention_source = "existing_attention_focal_model"
            self.attention_manifest = describe_p3_attention(model)
            return model

        model = _build_focal_model(self, cfg=cfg, weights=weights, verbose=verbose)
        self.attention_manifest = [install_p3_attention(model)]
        self.attention_source = "focal_model_after_pretrained_load"
        if count_p3_attention_wrappers(model) != 1:
            raise RuntimeError("expected exactly one P3 attention wrapper")
        return model
