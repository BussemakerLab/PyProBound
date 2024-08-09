"""Layer wrapping arbitrary modules.

Members are explicitly re-exported in pyprobound.layers.
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar, cast, overload

import torch
from torch import Tensor
from typing_extensions import Self, override

from ..alphabets import Alphabet
from ..table import Table
from .layer import Layer, LayerSpec

T = TypeVar("T", int, Tensor)


class ModuleSpec(LayerSpec):
    r"""Specification for any torch.nn.Module.

    The forward implementation of the module must take a sequence tensor of
    shape :math:`(\text{minibatch},\text{in_channels},\text{in_length})` and
    return a tensor of shape :math:`(\text{minibatch},1,1)`.
    """

    def __init__(
        self, alphabet: Alphabet, module: torch.nn.Module, name: str = ""
    ) -> None:
        """Initializes the Module specification.

        Args:
            alphabet: The alphabet used to encode sequences into tensors.
            module: The module used for scoring.
            name: A string used to describe the Module specification.
        """
        super().__init__(
            out_channels=1, in_channels=len(alphabet.alphabet), name=name
        )
        self._layers: set[ModuleLayer]  # type: ignore[assignment]
        self.module = module
        self.alphabet = alphabet

    @override
    def unfreeze(self, parameter: LayerSpec.unfreezable = "all") -> None:
        if parameter == "all":
            for p in self.module.parameters():
                p.requires_grad_()

    @override
    def __repr__(self) -> str:
        return repr(self.module)

    @override
    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        del mode
        return length * 0 + 1

    @overload
    def in_len(self, length: T, mode: Literal["min"]) -> T: ...

    @overload
    def in_len(self, length: T, mode: Literal["max"]) -> None: ...

    @override
    def in_len(
        self, length: T, mode: Literal["min", "max"] = "max"
    ) -> T | None:
        if mode == "min":
            return length * 0 + 1
        return None


class ModuleLayer(Layer):
    r"""Layer wrapper for torch.nn.Module.

    The forward implementation of the module must take a sequence tensor of
    shape :math:`(\text{minibatch},\text{in_channels},\text{in_length})` and
    return a tensor of shape :math:`(\text{minibatch},1,1)`.
    """

    def __init__(
        self,
        layer_spec: ModuleSpec,
        input_shape: int,
        min_input_length: int,
        max_input_length: int,
    ) -> None:
        """Initializes the Module layer.

        Args:
            layer_spec: The specification of the Module layer.
            input_shape: The number of elements in an input sequence.
            min_input_length: The minimum number of finite elements in an input
                sequence.
            max_input_length: The maximum number of finite elements in an input
                sequence.
        """
        super().__init__(
            layer_spec=layer_spec,
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
        )
        self.layer_spec: ModuleSpec

    @classmethod
    def from_spec(cls, spec: ModuleSpec, prev: Table[Any] | Layer) -> Self:
        """Creates a new instance from a specification and an input component.

        Args:
            spec: The specification of the Module layer.
            prev: If used as the first layer, the table that will be passed as
                an input; otherwise, the layer that precedes it.
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
        )

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        r"""Applies the module.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{in_length})`.

        Returns:
            A tensor of shape :math:`(\text{minibatch},1,1)`.
        """
        if seqs.ndim == 2:
            seqs = self.layer_spec.alphabet.embed(seqs)
        return cast(Tensor, self.layer_spec.module(seqs))
