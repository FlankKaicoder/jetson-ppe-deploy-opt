"""Reusable neural-network blocks for project experiments."""

from .attention_block import (
    P3AttentionWrapper,
    ResidualCBAMLite,
    count_p3_attention_wrappers,
    describe_p3_attention,
    install_p3_attention,
)

__all__ = [
    "P3AttentionWrapper",
    "ResidualCBAMLite",
    "count_p3_attention_wrappers",
    "describe_p3_attention",
    "install_p3_attention",
]
