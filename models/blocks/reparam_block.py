from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Sequential):
    """Convolution followed by BatchNorm, without activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )


class RepConvBlock(nn.Module):
    """
    Training form:
        3x3 Conv-BN
        + 1x1 Conv-BN
        + optional Identity-BN
        -> sum -> activation

    Deployment form:
        one 3x3 Conv with bias -> activation

    Exp04.0 intentionally fixes groups=1. More complicated grouped or
    depth-wise variants will not be introduced before the basic fusion
    path has been verified.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        deploy: bool = False,
        activation: bool = True,
    ) -> None:
        super().__init__()

        if stride not in (1, 2):
            raise ValueError(
                f"unsupported stride: {stride}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.deploy = deploy

        self.activation = (
            nn.SiLU()
            if activation
            else nn.Identity()
        )

        if deploy:
            self.reparam_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True,
            )
        else:
            self.branch_3x3 = ConvBN(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            )

            self.branch_1x1 = ConvBN(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )

            self.branch_identity = (
                nn.BatchNorm2d(in_channels)
                if (
                    in_channels == out_channels
                    and stride == 1
                )
                else None
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.deploy:
            return self.activation(
                self.reparam_conv(x)
            )

        output = (
            self.branch_3x3(x)
            + self.branch_1x1(x)
        )

        if self.branch_identity is not None:
            output = (
                output
                + self.branch_identity(x)
            )

        return self.activation(output)

    @staticmethod
    def _fuse_conv_bn(
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse a Conv2d and BatchNorm2d.

        W_fused = W * gamma / sqrt(var + eps)

        b_fused =
            beta
            + (b_conv - running_mean)
            * gamma / sqrt(var + eps)
        """

        kernel = conv.weight

        if conv.bias is None:
            conv_bias = torch.zeros(
                conv.out_channels,
                device=kernel.device,
                dtype=kernel.dtype,
            )
        else:
            conv_bias = conv.bias

        standard_deviation = torch.sqrt(
            bn.running_var + bn.eps
        )

        scale = (
            bn.weight
            / standard_deviation
        )

        fused_kernel = (
            kernel
            * scale.reshape(-1, 1, 1, 1)
        )

        fused_bias = (
            bn.bias
            + (
                conv_bias
                - bn.running_mean
            )
            * scale
        )

        return fused_kernel, fused_bias

    @staticmethod
    def _fuse_identity_bn(
        bn: nn.BatchNorm2d,
        channels: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Represent identity mapping as a 3x3 convolution.

        For every channel, the center element of the corresponding
        convolution kernel is 1 and all other elements are 0.
        """

        kernel = torch.zeros(
            (
                channels,
                channels,
                3,
                3,
            ),
            device=bn.weight.device,
            dtype=bn.weight.dtype,
        )

        index = torch.arange(
            channels,
            device=bn.weight.device,
        )

        kernel[
            index,
            index,
            1,
            1,
        ] = 1.0

        standard_deviation = torch.sqrt(
            bn.running_var + bn.eps
        )

        scale = (
            bn.weight
            / standard_deviation
        )

        fused_kernel = (
            kernel
            * scale.reshape(-1, 1, 1, 1)
        )

        fused_bias = (
            bn.bias
            - bn.running_mean
            * scale
        )

        return fused_kernel, fused_bias

    def get_equivalent_kernel_bias(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            return (
                self.reparam_conv.weight,
                self.reparam_conv.bias,
            )

        kernel_3x3, bias_3x3 = (
            self._fuse_conv_bn(
                self.branch_3x3[0],
                self.branch_3x3[1],
            )
        )

        kernel_1x1, bias_1x1 = (
            self._fuse_conv_bn(
                self.branch_1x1[0],
                self.branch_1x1[1],
            )
        )

        # Pad the 1x1 kernel to the center of a 3x3 kernel.
        kernel_1x1 = F.pad(
            kernel_1x1,
            (1, 1, 1, 1),
        )

        if self.branch_identity is None:
            kernel_identity = (
                torch.zeros_like(
                    kernel_3x3
                )
            )

            bias_identity = (
                torch.zeros_like(
                    bias_3x3
                )
            )
        else:
            (
                kernel_identity,
                bias_identity,
            ) = self._fuse_identity_bn(
                self.branch_identity,
                self.in_channels,
            )

        equivalent_kernel = (
            kernel_3x3
            + kernel_1x1
            + kernel_identity
        )

        equivalent_bias = (
            bias_3x3
            + bias_1x1
            + bias_identity
        )

        return (
            equivalent_kernel,
            equivalent_bias,
        )

    def switch_to_deploy(
        self,
    ) -> "RepConvBlock":
        """
        Replace all training branches with one equivalent 3x3 Conv.

        Calling the method repeatedly is safe.
        """

        if self.deploy:
            return self

        kernel, bias = (
            self.get_equivalent_kernel_bias()
        )

        reparam_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=True,
        ).to(
            device=kernel.device,
            dtype=kernel.dtype,
        )

        with torch.no_grad():
            reparam_conv.weight.copy_(kernel)
            reparam_conv.bias.copy_(bias)

        self.reparam_conv = reparam_conv

        del self.branch_3x3
        del self.branch_1x1
        del self.branch_identity

        self.deploy = True
        return self
