"""Maxpooling layer.

Members are explicitly re-exported in pyprobound.layers.
"""
from __future__ import annotations

from typing import Any, Literal, TypeVar

import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self, override

from ..table import Table
from ..utils import ceil_div, floor_div
from .layer import Layer, LayerSpec

T = TypeVar("T", int, Tensor)


class MaxPool1dSpec(LayerSpec):
    """Specification passed to torch.nn.MaxPool1d."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        ceil_mode: bool = True,
        name: str = "",
    ) -> None:
        """Initializes the maxpooling specification.

        Args:
            in_channels: The number of input channels.
            kernel_size: The size of the sliding window.
            ceil_mode: Whether to use `ceil` instead of `floor` for the output
                shape.
            name: A string used to describe the maxpooling specification.
        """
        super().__init__(
            out_channels=in_channels, in_channels=in_channels, name=name
        )
        self._layers: set[MaxPool1d]  # type: ignore[assignment]
        self._kernel_size = kernel_size
        self._ceil_mode = ceil_mode

    @property
    def kernel_size(self) -> int:
        """The size of the sliding window."""
        return self._kernel_size

    @property
    def ceil_mode(self) -> bool:
        """Whether to use `ceil` instead of `floor` for the output shape."""
        return self._ceil_mode

    @override
    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        if mode == "min":
            return ceil_div(length, self.kernel_size)
        if mode == "max":
            return ceil_div(length - 1, self.kernel_size) + 1
        if self.ceil_mode:
            return ceil_div(length, self.kernel_size)
        return floor_div(length, self.kernel_size)

    @override
    def in_len(self, length: T, mode: Literal["min", "max"] = "max") -> T:
        if self.ceil_mode:
            if mode == "min":
                return self.kernel_size * (length - 1) + 1
            return self.kernel_size * length
        if mode == "min":
            return self.kernel_size * length
        return self.kernel_size * (length + 1) - 1


class MaxPool1d(Layer):
    """Layer wrapper for torch.nn.MaxPool1d."""

    def __init__(
        self,
        layer_spec: MaxPool1dSpec,
        input_shape: int,
        min_input_length: int,
        max_input_length: int,
        name: str = "",
    ) -> None:
        """Initializes the maxpooling layer.

        Args:
            layer_spec: The specification of the maxpooling layer.
            input_shape: The number of elements in an input sequence.
            min_input_length: The minimum number of finite elements in an input
                sequence.
            max_input_length: The maximum number of finite elements in an input
                sequence.
            name: A string used to describe the maxpooling layer.
        """
        super().__init__(
            layer_spec=layer_spec,
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
            name=name,
        )
        self.layer_spec: MaxPool1dSpec

    @classmethod
    def from_spec(
        cls, spec: MaxPool1dSpec, prev: Table[Any] | Layer, name: str = ""
    ) -> Self:
        """Creates a new instance from a specification and an input component.

        Args:
            spec: The specification of the maxpooling layer.
            prev: If used as the first layer, the table that will be passed as
                an input; otherwise, the layer that precedes it.
            name: A string used to describe the maxpooling layer.
        """
        if isinstance(prev, Layer):
            input_shape = prev.out_len(prev.input_shape, "shape")
            min_input_length = prev.out_len(prev.min_input_length, "min")
            max_input_length = prev.out_len(prev.max_input_length, "max")
        else:
            input_shape = prev.input_shape
            min_input_length = prev.min_read_length
            max_input_length = prev.max_read_length
        return cls(
            layer_spec=spec,
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
            name=name,
        )

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        r"""Applies a maxpooling over the last dimension.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{channels},\text{length})`.

        Returns:
            The maxpooled tensor of shape
            :math:`(\text{minibatch},\text{channels},`
            :math:`\text{ceil_mode ? ceil : floor}
            (\text{length} / \text{kernel_size}))`.
        """
        return F.max_pool1d(
            seqs,
            self.layer_spec.kernel_size,
            ceil_mode=self.layer_spec.ceil_mode,
        )
