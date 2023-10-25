"""Roll layer for changing between left, right, and center padding.

Members are explicitly re-exported in pyprobound.layers.
"""
from typing import Any, Literal, TypeVar, cast, overload

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Self, override

from .. import __precision__
from ..alphabets import Alphabet
from ..table import Table
from .layer import Layer, LayerSpec

T = TypeVar("T", int, Tensor)


class RollSpec(LayerSpec):
    """Specification for changing between left, right, and center padding."""

    def __init__(
        self,
        alphabet: Alphabet,
        direction: Literal["left", "right", "center"],
        max_length: int | None = None,
        include_n: bool = False,
        name: str = "",
    ) -> None:
        """Initializes the rolling specification.

        Args:
            alphabet: The alphabet used to encode sequences into tensors.
            direction: Whether to justify the ends of sequences on the left,
                right, or center.
            max_length: The number of elements to keep from the aligned end.
                Ex. `[...,:max_length]` if left, `[...,-max_length:]` if right.
            include_n: Whether to use the `*` character as a padding value.
            name: A string used to describe the rolling specification.
        """
        if max_length is not None and direction == "center":
            raise ValueError("Cannot specify max_length with center padding.")
        super().__init__(
            out_channels=len(alphabet.alphabet),
            in_channels=len(alphabet.alphabet),
            name=name,
        )
        self.alphabet = alphabet
        self._layers: set[Roll]  # type: ignore[assignment]
        self.direction = direction
        self.max_length = max_length
        self.include_n = include_n

    @override
    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        if self.max_length is not None:
            if mode == "shape":
                return length * 0 + self.max_length
            return cast(T, np.minimum(length, self.max_length))
        return super().out_len(length, mode)

    @overload
    def in_len(self, length: T, mode: Literal["min"]) -> T:
        ...

    @overload
    def in_len(self, length: T, mode: Literal["max"]) -> None:
        ...

    @override
    def in_len(
        self, length: T, mode: Literal["min", "max"] = "max"
    ) -> T | None:
        if self.max_length is None:
            return super().in_len(length, mode)
        if mode == "min":
            return cast(T, np.minimum(length, self.max_length))
        return None


class Roll(Layer):
    """Layer for changing between left, right, and center padding.

    Assumes left-padded input.
    """

    def __init__(
        self,
        layer_spec: RollSpec,
        input_shape: int,
        min_input_length: int,
        max_input_length: int,
        name: str = "",
    ) -> None:
        """Initializes the rolling layer.

        Args:
            layer_spec: The specification of the rolling layer.
            input_shape: The number of elements in an input sequence.
            min_input_length: The minimum number of finite elements in an input
                sequence.
            max_input_length: The maximum number of finite elements in an input
                sequence.
            name: A string used to describe the roll layer.
        """
        super().__init__(
            layer_spec=layer_spec,
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
            name=name,
        )
        self.layer_spec: RollSpec

    @classmethod
    def from_spec(
        cls, spec: RollSpec, prev: Table[Any] | Layer, name: str = ""
    ) -> Self:
        """Creates a new instance from a specification and an input component.

        Args:
            spec: The specification of the roll layer.
            prev: If used as the first layer, the table that will be passed as
                an input; otherwise, the layer that precedes it.
            name: A string used to describe the roll layer.
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
        r"""Justifies the sequences to the specified end.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{channels},\text{length})`.

        Returns:
            The re-justified tensor with same dimensions as the input, of shape
            :math:`(\ldots,\text{max_length is None ? length : max_length})`.
        """
        if self.layer_spec.direction != "left":
            # Get the number of elements to shift each row by
            if seqs.ndim == 2:
                alphabet = self.layer_spec.alphabet
                pad_vals = [
                    " " * alphabet.monomer_length,
                    "-" * alphabet.monomer_length,
                ]
                if self.layer_spec.include_n:
                    pad_vals.append("*" * alphabet.monomer_length)
                indices = [alphabet.get_index[c] for c in pad_vals]
                roll = cast(
                    Tensor, sum((seqs == idx).sum(dim=1) for idx in indices)
                )
            else:
                roll = (~seqs.sum(1).isfinite()).sum(1)

            # Adjust for center padding
            if self.layer_spec.direction == "center":
                roll //= 2

            # Implement vectorized shift with gather
            indexing = (
                torch.arange(seqs.shape[-1]).unsqueeze(0) - roll.unsqueeze(1)
            ) % seqs.shape[-1]
            if seqs.ndim == 3:
                indexing = indexing.unsqueeze(1).expand(-1, seqs.shape[-2], -1)
            out = torch.gather(seqs, -1, indexing)
        else:
            out = seqs

        # Trim to max_length
        if self.layer_spec.max_length is not None:
            if self.layer_spec.direction == "left":
                return out[..., : self.layer_spec.max_length]
            return out[..., -self.layer_spec.max_length :]
        return out
