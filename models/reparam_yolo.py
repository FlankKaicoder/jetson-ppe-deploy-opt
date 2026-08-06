from __future__ import annotations

import copy
from typing import Any, Iterable

import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules import Conv as UltralyticsConv
except ImportError:
    from ultralytics.nn.modules.conv import Conv as UltralyticsConv

from models.blocks.reparam_block import RepConvBlock


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])

    if isinstance(value, list):
        return int(value[0]), int(value[1])

    return int(value), int(value)


def is_neck_downsample_candidate(
    module: nn.Module,
) -> bool:
    """
    Select a top-level Ultralytics Conv that can be replaced one-for-one
    by the current RepConvBlock implementation.

    Conditions:
        3x3 kernel
        stride 2
        padding 1
        dilation 1
        groups 1
        input channels == output channels
    """

    if not isinstance(module, UltralyticsConv):
        return False

    conv = module.conv

    return (
        _pair(conv.kernel_size) == (3, 3)
        and _pair(conv.stride) == (2, 2)
        and _pair(conv.padding) == (1, 1)
        and _pair(conv.dilation) == (1, 1)
        and conv.groups == 1
        and conv.in_channels == conv.out_channels
    )


def find_neck_downsample_indices(
    detection_model: nn.Module,
    minimum_index: int = 11,
) -> list[int]:
    """
    Search only the YOLO Head/Neck region.

    For the current YOLO11n structure, the expected result is [17, 20].
    The result is discovered from the real loaded graph rather than
    assumed blindly.
    """

    if not hasattr(detection_model, "model"):
        raise TypeError(
            "detection_model does not contain a top-level .model graph"
        )

    indices: list[int] = []

    for index, module in enumerate(detection_model.model):
        if index < minimum_index:
            continue

        if is_neck_downsample_candidate(module):
            indices.append(index)

    return indices


def _copy_batch_norm(
    source: nn.BatchNorm2d,
    destination: nn.BatchNorm2d,
) -> None:
    if source.num_features != destination.num_features:
        raise ValueError(
            "BatchNorm feature count mismatch: "
            f"{source.num_features} != {destination.num_features}"
        )

    with torch.no_grad():
        destination.weight.copy_(source.weight)
        destination.bias.copy_(source.bias)
        destination.running_mean.copy_(
            source.running_mean
        )
        destination.running_var.copy_(
            source.running_var
        )
        destination.num_batches_tracked.copy_(
            source.num_batches_tracked
        )

    destination.eps = source.eps
    destination.momentum = source.momentum


def _copy_ultralytics_metadata(
    source: nn.Module,
    destination: nn.Module,
) -> None:
    """
    Ultralytics BaseModel forward uses module.i and module.f.

    Losing these attributes when replacing a top-level module would make
    the graph forward fail even when the tensor computation is correct.
    """

    for attribute in ("i", "f"):
        if hasattr(source, attribute):
            setattr(
                destination,
                attribute,
                copy.deepcopy(
                    getattr(source, attribute)
                ),
            )

    destination.type = (
        f"{RepConvBlock.__module__}."
        f"{RepConvBlock.__name__}"
    )

    destination.np = sum(
        parameter.numel()
        for parameter in destination.parameters()
    )


def build_repconv_from_ultralytics_conv(
    source: UltralyticsConv,
) -> RepConvBlock:
    """
    Convert an already initialized Ultralytics Conv-BN-Activation block
    into the training form of RepConvBlock.

    Initialization policy:
        original Conv-BN -> Rep 3x3 branch
        new 1x1 branch  -> zero BN gamma and beta
        optional identity branch -> zero BN gamma and beta

    Therefore the new branches initially contribute zero, and the whole
    RepConvBlock initially reproduces the original Conv output.
    """

    if not is_neck_downsample_candidate(source):
        raise ValueError(
            "source module does not satisfy the current "
            "RepConv replacement constraints"
        )

    source_conv = source.conv
    source_bn = source.bn

    if source_conv.bias is not None:
        raise ValueError(
            "expected Ultralytics Conv without convolution bias"
        )

    stride = int(
        _pair(source_conv.stride)[0]
    )

    rep_block = RepConvBlock(
        in_channels=source_conv.in_channels,
        out_channels=source_conv.out_channels,
        stride=stride,
        deploy=False,
        activation=False,
    )

    rep_block = rep_block.to(
        device=source_conv.weight.device,
        dtype=source_conv.weight.dtype,
    )

    rep_block.activation = copy.deepcopy(
        source.act
    ).to(
        device=source_conv.weight.device
    )

    with torch.no_grad():
        rep_block.branch_3x3[0].weight.copy_(
            source_conv.weight
        )

    _copy_batch_norm(
        source_bn,
        rep_block.branch_3x3[1],
    )

    # The new branch exists structurally but contributes exactly zero at
    # initialization. Its BN scale can subsequently learn during training.
    with torch.no_grad():
        rep_block.branch_1x1[1].weight.zero_()
        rep_block.branch_1x1[1].bias.zero_()
        rep_block.branch_1x1[1].running_mean.zero_()
        rep_block.branch_1x1[1].running_var.fill_(1.0)
        rep_block.branch_1x1[1].num_batches_tracked.zero_()

        if rep_block.branch_identity is not None:
            rep_block.branch_identity.weight.zero_()
            rep_block.branch_identity.bias.zero_()
            rep_block.branch_identity.running_mean.zero_()
            rep_block.branch_identity.running_var.fill_(1.0)
            rep_block.branch_identity.num_batches_tracked.zero_()

    _copy_ultralytics_metadata(
        source,
        rep_block,
    )

    rep_block.train(
        source.training
    )

    return rep_block


def replace_neck_downsample_convs(
    detection_model: nn.Module,
    indices: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Replace selected top-level YOLO Conv modules in-place.

    Returns a manifest describing every replaced module.
    """

    if indices is None:
        selected_indices = (
            find_neck_downsample_indices(
                detection_model
            )
        )
    else:
        selected_indices = [
            int(index)
            for index in indices
        ]

    manifest: list[dict[str, Any]] = []

    for index in selected_indices:
        source = detection_model.model[index]

        if not isinstance(
            source,
            UltralyticsConv,
        ):
            raise TypeError(
                f"model layer {index} is not an Ultralytics Conv: "
                f"{source.__class__.__name__}"
            )

        source_conv = source.conv

        rep_block = (
            build_repconv_from_ultralytics_conv(
                source
            )
        )

        detection_model.model[index] = (
            rep_block
        )

        manifest.append(
            {
                "index": index,
                "source_class": (
                    source.__class__.__name__
                ),
                "replacement_class": (
                    rep_block.__class__.__name__
                ),
                "in_channels": (
                    source_conv.in_channels
                ),
                "out_channels": (
                    source_conv.out_channels
                ),
                "kernel_size": list(
                    _pair(
                        source_conv.kernel_size
                    )
                ),
                "stride": list(
                    _pair(
                        source_conv.stride
                    )
                ),
                "padding": list(
                    _pair(
                        source_conv.padding
                    )
                ),
                "groups": source_conv.groups,
                "source_parameter_count": sum(
                    parameter.numel()
                    for parameter
                    in source.parameters()
                ),
                "training_parameter_count": sum(
                    parameter.numel()
                    for parameter
                    in rep_block.parameters()
                ),
            }
        )

    return manifest


def count_reparam_blocks(
    module: nn.Module,
) -> int:
    return sum(
        1
        for item in module.modules()
        if isinstance(
            item,
            RepConvBlock,
        )
    )


def switch_reparam_blocks_to_deploy(
    module: nn.Module,
) -> int:
    """
    Fuse all RepConvBlock instances in-place.

    The list is materialized first so that changing a block's child
    modules during conversion does not affect traversal.
    """

    blocks = [
        item
        for item in module.modules()
        if isinstance(
            item,
            RepConvBlock,
        )
    ]

    for block in blocks:
        block.switch_to_deploy()

    return len(blocks)


def all_reparam_blocks_deployed(
    module: nn.Module,
) -> bool:
    blocks = [
        item
        for item in module.modules()
        if isinstance(
            item,
            RepConvBlock,
        )
    ]

    return bool(blocks) and all(
        block.deploy
        for block in blocks
    )
