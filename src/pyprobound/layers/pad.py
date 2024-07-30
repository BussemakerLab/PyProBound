"""Padding layer.

Members are explicitly re-exported in pyprobound.layers.
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self, override

from ..alphabets import Alphabet
from ..table import Table
from .layer import Layer, LayerSpec
from .roll import Roll, RollSpec

T = TypeVar("T", int, Tensor)


class PadSpec(LayerSpec):
    """Specification passed to F.pad."""

    def __init__(
        self, alphabet: Alphabet, left: int = 0, right: int = 0, name: str = ""
    ) -> None:
        """Initializes the padding specification.

        Args:
            alphabet: The alphabet used to encode sequences into tensors.
            left: The number of elements to pad on the left.
            right: The number of elements to pad on the right.
            name: A string used to describe the padding specification.
        """
        super().__init__(
            out_channels=len(alphabet.alphabet),
            in_channels=len(alphabet.alphabet),
            name=name,
        )
        self._layers: set[Pad]  # type: ignore[assignment]
        self._left = left
        self._right = right
        self.value = alphabet.wildcard_pad
        self.alphalen = len(alphabet.alphabet)

    @property
    def left(self) -> int:
        """The number of elements to pad on the left."""
        return self._left

    @property
    def right(self) -> int:
        """The number of elements to pad on the right."""
        return self._right

    @override
    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        del mode
        return length + self.left + self.right

    @override
    def in_len(self, length: T, mode: Literal["min", "max"] = "max") -> T:
        del mode
        return length - self.left - self.right


class Pad(Layer):
    """Layer wrapper for F.pad."""

    def __init__(
        self,
        layer_spec: PadSpec,
        input_shape: int,
        min_input_length: int,
        max_input_length: int,
        name: str = "",
    ) -> None:
        """Initializes the padding layer.

        Args:
            layer_spec: The specification of the padding layer.
            input_shape: The number of elements in an input sequence.
            min_input_length: The minimum number of finite elements in an input
                sequence.
            max_input_length: The maximum number of finite elements in an input
                sequence.
            name: A string used to describe the padding layer.
        """
        super().__init__(
            layer_spec=layer_spec,
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
            name=name,
        )
        self.layer_spec: PadSpec

    @classmethod
    def from_spec(
        cls, spec: PadSpec, prev: Table[Any] | Layer, name: str = ""
    ) -> Self:
        """Creates a new instance from a specification and an input component.

        Args:
            spec: The specification of the padding layer.
            prev: If used as the first layer, the table that will be passed as
                an input; otherwise, the layer that precedes it.
            name: A string used to describe the padding layer.
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
        r"""Applies a padding over the last dimension.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{channels},\text{length})`.

        Returns:
            The maxpooled tensor of shape
            :math:`(\text{minibatch},\text{channels},`
            :math:`\text{ceil_mode ? ceil : floor}
            (\text{length} / \text{kernel_size}))`.
        """
        if seqs.ndim == 3:
            value = 1 / self.layer_spec.alphalen
        else:
            value = self.layer_spec.value

        return F.pad(
            seqs, (self.layer_spec.left, self.layer_spec.right), value=value
        )


def get_padding_layers(
    alphabet: Alphabet, prev: Table[Any] | Layer, left: int = 0, right: int = 0
) -> list[Layer]:
    """Layers for padding, with adjustment for variable length sequences.

    Assumes input is left-aligned.

    Args:
        alphabet: The alphabet used to encode sequences into tensors.
        prev: If used as the first layer, the table that will be passed as
            an input; otherwise, the layer that precedes it.
        left: The number of elements to pad on the left.
        right: The number of elements to pad on the right.

    Returns:
        The layers needed to pad sequences with variable length sequences.
    """
    layers: list[Layer] = []
    if left != 0:
        layers.append(
            Pad.from_spec(spec=PadSpec(alphabet, left=left), prev=prev)
        )
        prev = layers[-1]
    if right != 0:
        layers.append(
            Roll.from_spec(
                RollSpec(alphabet=alphabet, direction="right"), prev=prev
            )
        )
        layers.append(
            Pad.from_spec(spec=PadSpec(alphabet, right=right), prev=layers[-1])
        )
        layers.append(
            Roll.from_spec(
                RollSpec(alphabet=alphabet, direction="left"), prev=layers[-1]
            )
        )
    return layers
