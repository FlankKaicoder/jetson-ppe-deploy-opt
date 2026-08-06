from __future__ import annotations

import copy
from typing import Any

import torch.nn as nn

try:
    from ultralytics.models.yolo.detect import (
        DetectionTrainer,
    )
except ImportError:
    from ultralytics.models.yolo.detect.train import (
        DetectionTrainer,
    )

from models.blocks.reparam_block import RepConvBlock
from models.reparam_yolo import (
    count_reparam_blocks,
    find_neck_downsample_indices,
    replace_neck_downsample_convs,
)


def describe_existing_reparam_model(
    model: nn.Module,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []

    if not hasattr(model, "model"):
        return manifest

    for index, module in enumerate(model.model):
        if not isinstance(module, RepConvBlock):
            continue

        manifest.append(
            {
                "index": index,
                "source_class": (
                    "existing_checkpoint"
                ),
                "replacement_class": (
                    module.__class__.__name__
                ),
                "in_channels": (
                    module.in_channels
                ),
                "out_channels": (
                    module.out_channels
                ),
                "stride": (
                    module.stride
                ),
                "deploy": (
                    module.deploy
                ),
            }
        )

    return manifest


class ReparamDetectionTrainer(
    DetectionTrainer
):
    """
    DetectionTrainer that replaces the two YOLO11 Neck downsampling
    Conv-BN blocks with RepConvBlock after normal pretrained-weight
    loading.

    Initial COCO training path:
        build normal YOLO11
        -> load COCO weights
        -> replace layers 17 and 20
        -> copy original Conv-BN into the 3x3 branches
        -> initialize new 1x1 branches with zero contribution
    """

    def get_model(
        self,
        cfg: Any = None,
        weights: Any = None,
        verbose: bool = True,
    ) -> nn.Module:
        # Preserve an already constructed Rep model when a future
        # checkpoint/resume path supplies the actual model object.
        if (
            isinstance(weights, nn.Module)
            and count_reparam_blocks(weights) > 0
        ):
            model = copy.deepcopy(weights)

            self.reparam_source = (
                "existing_reparam_model"
            )

            self.reparam_manifest = (
                describe_existing_reparam_model(
                    model
                )
            )

            return model

        model = super().get_model(
            cfg=cfg,
            weights=weights,
            verbose=verbose,
        )

        indices = (
            find_neck_downsample_indices(
                model
            )
        )

        if indices != [17, 20]:
            raise RuntimeError(
                "unexpected YOLO11 Neck downsample "
                f"candidate indices: {indices}"
            )

        self.reparam_manifest = (
            replace_neck_downsample_convs(
                model,
                indices=indices,
            )
        )

        self.reparam_source = (
            "normal_model_after_pretrained_load"
        )

        if count_reparam_blocks(model) != 2:
            raise RuntimeError(
                "expected two RepConvBlock modules "
                "after replacement"
            )

        return model
