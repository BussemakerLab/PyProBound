"""Layers"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, TypeVar, cast, overload

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self, override

from .alphabet import Alphabet
from .base import BindingOptim, Component, Spec, Transform
from .containers import Buffer
from .table import Table
from .utils import ceil_div, floor_div

T = TypeVar("T", int, Tensor)
BindingMode = Any


class LayerSpec(Spec):
    """Stores experiment-independent parameters of a Layer

    Attributes:
        _layers:
            Set containing all Layers a Param appears
    """

    def __init__(
        self, out_channels: int, in_channels: int, name: str = ""
    ) -> None:
        super().__init__(name=name)
        self._out_channels: Tensor = Buffer(torch.tensor(out_channels))
        self._in_channels: Tensor = Buffer(torch.tensor(in_channels))
        self._layers: set[Layer] = set()

    @property
    def out_channels(self) -> int:
        return cast(int, self._out_channels.item())

    @property
    def in_channels(self) -> int:
        return cast(int, self._in_channels.item())

    @override
    def components(self) -> Iterator[Component]:
        return iter(())

    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        """Output length"""
        del mode
        return length

    @overload
    def in_len(self, length: T, mode: Literal["min"]) -> T:
        ...

    @overload
    def in_len(self, length: T, mode: Literal["max"]) -> T | None:
        ...

    def in_len(
        self, length: T, mode: Literal["min", "max"] = "max"
    ) -> T | None:
        """Receptive field"""
        del mode
        return length


class BindingModeKey(tuple[LayerSpec]):
    """Output of BindingMode.key() with length calculations"""

    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        """Output length"""
        for layer in iter(self):
            length = layer.out_len(length, mode=mode)
        return length

    @overload
    def in_len(self, length: T, mode: Literal["min"]) -> T:
        ...

    @overload
    def in_len(self, length: T, mode: Literal["max"]) -> T | None:
        ...

    def in_len(
        self, length: T, mode: Literal["min", "max"] = "max"
    ) -> T | None:
        """Receptive field"""
        for layer in reversed(self):
            out = layer.in_len(length, mode=mode)
            if out is None:
                return out
            length = out
        return length


class Layer(Transform):
    """Experiment-specific Spec container used sequentially in BindingMode

    Attributes:
        _binding_mode:
            Set containing tuples of every BindingMode and respective index
            that a Layer appears in.
    """

    def __init__(
        self,
        layer_spec: LayerSpec,
        input_shape: int,
        min_input_length: int,
        max_input_length: int,
    ) -> None:
        super().__init__()
        self.layer_spec = layer_spec
        self.layer_spec._layers.add(self)
        self._input_shape: Tensor = Buffer(torch.tensor(input_shape))
        self._min_input_length: Tensor = Buffer(torch.tensor(min_input_length))
        self._max_input_length: Tensor = Buffer(torch.tensor(max_input_length))
        self._binding_mode: set[tuple["BindingMode", int]] = set()

    @property
    def input_shape(self) -> int:
        return cast(int, self._input_shape.item())

    @property
    def min_input_length(self) -> int:
        return cast(int, self._min_input_length.item())

    @property
    def max_input_length(self) -> int:
        return cast(int, self._max_input_length.item())

    @property
    def out_channels(self) -> int:
        return self.layer_spec.out_channels

    @property
    def in_channels(self) -> int:
        return self.layer_spec.in_channels

    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        """Output length"""
        return self.layer_spec.out_len(length=length, mode=mode)

    @overload
    def in_len(self, length: T, mode: Literal["min"]) -> T:
        ...

    @overload
    def in_len(self, length: T, mode: Literal["max"]) -> T | None:
        ...

    def in_len(
        self, length: T, mode: Literal["min", "max"] = "max"
    ) -> T | None:
        """Receptive field"""
        return self.layer_spec.in_len(length=length, mode=mode)

    def update_binding_optim(
        self, binding_optim: BindingOptim
    ) -> BindingOptim:
        return binding_optim

    @override
    def check_length_consistency(self) -> None:
        if (
            not 1
            <= self.min_input_length
            <= self.input_shape
            <= self.max_input_length
        ):
            raise RuntimeError(
                f"The attribute ordering does not hold:"
                f" 1 <= min_input_length={self.min_input_length}"
                f"  <= input_shape={self.input_shape}"
                f"  <= max_input_length={self.max_input_length}"
            )
        if self.out_len(self.min_input_length, mode="min") < 1:
            raise ValueError("Minimum output length is less than 1")

    @override
    def components(self) -> Iterator[Component]:
        yield self.layer_spec

    def lengths(self, seqs: Tensor) -> Tensor:
        """Number of monomers in each sequence"""
        if len(seqs.shape) == 2:
            alphabet: Alphabet | None = getattr(
                self.layer_spec, "alphabet", None
            )
            if alphabet is None:
                raise ValueError(
                    "sequences not embedded but alphabet is not specified"
                )
            return torch.sum(seqs != alphabet.neginf_pad, dim=1)
        return torch.sum(seqs, dim=1).isfinite().sum(dim=1)

    def update_input_length(
        self,
        left_shift: int = 0,
        right_shift: int = 0,
        min_len_shift: int = 0,
        max_len_shift: int = 0,
        new_min_len: int | None = None,
        new_max_len: int | None = None,
    ) -> None:
        self._input_shape += left_shift + right_shift
        if new_min_len is None:
            new_min_len = (
                self.min_input_length
                + min_len_shift
                + left_shift
                + right_shift
            )
        if new_max_len is None:
            new_max_len = (
                self.max_input_length
                + max_len_shift
                + left_shift
                + right_shift
            )
        if new_max_len - new_min_len != (
            self.max_input_length + max_len_shift
        ) - (self.min_input_length + min_len_shift):
            raise ValueError(
                f"New min/max len {(new_max_len, new_min_len)} incompatible"
                f" with min/max shift {(min_len_shift, max_len_shift)}"
            )
        self._min_input_length += new_min_len - self.min_input_length
        self._max_input_length += new_max_len - self.max_input_length


class MaxPool1dSpec(LayerSpec):
    """LayerSpec wrapper around torch.nn.MaxPool1d"""

    _layers: set[MaxPool1d]  # type: ignore[assignment]

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        ceil_mode: bool = True,
        name: str = "",
    ) -> None:
        super().__init__(
            out_channels=in_channels, in_channels=in_channels, name=name
        )
        self._kernel_size: Tensor = Buffer(torch.tensor(kernel_size))
        self._ceil_mode: Tensor = Buffer(torch.tensor(ceil_mode))

    @property
    def kernel_size(self) -> int:
        return cast(int, self._kernel_size.item())

    @property
    def ceil_mode(self) -> bool:
        return cast(bool, self._ceil_mode.item())

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
    """Wrapper around torch.nn.MaxPool1d"""

    layer_spec: MaxPool1dSpec

    def __init__(
        self,
        layer_spec: MaxPool1dSpec,
        input_shape: int,
        min_input_length: int,
        max_input_length: int,
    ) -> None:
        super().__init__(
            layer_spec=layer_spec,
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
        )

    @property
    def kernel_size(self) -> int:
        return self.layer_spec.kernel_size

    @property
    def ceil_mode(self) -> bool:
        return self.layer_spec.ceil_mode

    @classmethod
    def from_spec(cls, spec: MaxPool1dSpec, prev: Table[Any] | Layer) -> Self:
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
        return cast(
            Tensor,
            F.max_pool1d(seqs, self.kernel_size, ceil_mode=self.ceil_mode),
        )


class Roll(Layer):
    """Center a sequence"""

    def __init__(
        self,
        alphabet: Alphabet,
        input_shape: int,
        min_input_length: int,
        max_input_length: int,
    ) -> None:
        super().__init__(
            LayerSpec(
                out_channels=len(alphabet.alphabet),
                in_channels=len(alphabet.alphabet),
            ),
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
        )
        self.layer_spec.alphabet = alphabet
        self.alphabet = alphabet

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        if len(seqs.shape) == 2:
            indices = [self.alphabet.get_index[c] for c in (" ", "-", "*")]
            return torch.stack(
                [
                    seq.roll(
                        sum(torch.sum(seq == idx).item() for idx in indices)
                        // 2,
                        dims=0,
                    )
                    for seq in seqs
                ]
            )
        return torch.stack(
            [
                seq.roll(
                    (~seq.sum(dim=0).isfinite()).sum(dim=-1).item() // 2,
                    dims=0,
                )
                for seq in seqs
            ]
        )
