"""Base classes for Mode layers.

Members are explicitly re-exported in pyprobound.layers.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, Protocol, TypeVar, cast, overload

import torch
from torch import Tensor
from typing_extensions import override

from ..alphabets import Alphabet
from ..base import BindingOptim, Component, Spec, Transform

T = TypeVar("T", int, Tensor)
Mode = Any


class LengthManager(Protocol):
    """Protocol for output length and receptive field calculations."""

    @property
    def out_channels(self) -> int:
        """The number of output channels."""

    @property
    def in_channels(self) -> int:
        """The number of input channels."""

    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        """Calculates the number of elements in the output length dimension.

        Args:
            length: The input length.
            mode: Either `shape`, which returns the number of elements, or
                `min` or `max`, which return the minimum or maximum number of
                finite elements.

        Returns:
            The number of elements in the output length dimension, according to
            the specified `mode`.
        """
        del mode
        return length

    @overload
    def in_len(self, length: T, mode: Literal["min"]) -> T: ...

    @overload
    def in_len(self, length: T, mode: Literal["max"]) -> T | None: ...

    def in_len(
        self, length: T, mode: Literal["min", "max"] = "max"
    ) -> T | None:
        """Calculates the receptive field.

        Args:
            length: The output length.
            mode: Either `min` or `max`, representing the minimum or maximum
                number of positions contributing to the output length.

        Returns:
            The number of input positions that contribute to the values of the
            corresponding number of output positions. Outputs None if the `max`
            receptive field is undefined.
        """
        del mode
        return length


class LayerSpec(Spec, LengthManager):
    """A component that stores experiment-independent parameters of layers.

    The forward implementation should be left to the Layer.

    Attributes:
        _layers (set[Layer]): All of the layers a specification appears in.
    """

    def __init__(
        self, out_channels: int, in_channels: int, name: str = ""
    ) -> None:
        """Initializes the layer specification.

        Args:
            out_channels: The number of output channels.
            in_channels: The number of input channels.
            name: A string used to describe the layer.
        """
        super().__init__(name=name)
        self._out_channels = out_channels
        self._in_channels = in_channels
        self._layers: set[Layer] = set()

    @override
    @property
    def out_channels(self) -> int:
        return self._out_channels

    @override
    @property
    def in_channels(self) -> int:
        return self._in_channels

    @override
    def components(self) -> Iterator[Component]:
        return iter(())


class EmptyLayerSpec(LayerSpec):
    """LayerSpec that does not require any configuration."""

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self))

    @override
    def __hash__(self) -> int:
        return hash(type(self))


class Layer(Transform, LengthManager):
    """Experiment-specific LayerSpec container.

    Attributes:
        layer_spec (LayerSpec): The specification of the layer.
        _modes (set[tuple[Mode, int]]): Tuples of every
            binding mode and respective index that a layer appears in.
    """

    def __init__(
        self,
        layer_spec: LayerSpec,
        input_shape: int,
        min_input_length: int,
        max_input_length: int,
        name: str = "",
    ) -> None:
        """Initializes the layer.

        Args:
            layer_spec: The specification of the layer.
            input_shape: The number of elements in an input sequence.
            min_input_length: The minimum number of finite elements in an input
                sequence.
            max_input_length: The maximum number of finite elements in an input
                sequence.
            name: A string used to describe the layer.
        """
        super().__init__(name=name)
        self.layer_spec = layer_spec
        self.layer_spec._layers.add(self)
        self._modes: set[tuple["Mode", int]] = set()
        self._input_shape: Tensor
        self.register_buffer("_input_shape", torch.tensor(input_shape))
        self._min_input_length: Tensor
        self.register_buffer(
            "_min_input_length", torch.tensor(min_input_length)
        )
        self._max_input_length: Tensor
        self.register_buffer(
            "_max_input_length", torch.tensor(max_input_length)
        )

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}( {repr(self.layer_spec)} )"

    @override
    def __str__(self) -> str:
        if self.layer_spec.name != "":
            return f"{type(self).__name__}-{self.layer_spec.name}"
        return self.__repr__()

    @override
    @property
    def out_channels(self) -> int:
        return self.layer_spec.out_channels

    @override
    @property
    def in_channels(self) -> int:
        return self.layer_spec.in_channels

    @property
    def input_shape(self) -> int:
        """The number of elements in an input sequence."""
        return cast(int, self._input_shape.item())

    @property
    def min_input_length(self) -> int:
        """The minimum number of finite elements in an input sequence."""
        return cast(int, self._min_input_length.item())

    @property
    def max_input_length(self) -> int:
        """The maximum number of finite elements in an input sequence."""
        return cast(int, self._max_input_length.item())

    @override
    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        return self.layer_spec.out_len(length=length, mode=mode)

    @overload
    def in_len(self, length: T, mode: Literal["min"]) -> T: ...

    @overload
    def in_len(self, length: T, mode: Literal["max"]) -> T | None: ...

    @override
    def in_len(
        self, length: T, mode: Literal["min", "max"] = "max"
    ) -> T | None:
        return self.layer_spec.in_len(length=length, mode=mode)

    @override
    def components(self) -> Iterator[Component]:
        yield self.layer_spec

    @override
    def check_length_consistency(self) -> None:
        if not 1 <= self.min_input_length <= self.max_input_length:
            raise RuntimeError(
                f"The ordering does not hold:"
                f" 1 <= min_input_length={self.min_input_length}"
                f"  <= max_input_length={self.max_input_length}"
            )
        if self.out_len(self.min_input_length, mode="min") < 1:
            raise ValueError("Minimum output length is less than 1")

    def update_binding_optim(
        self, binding_optim: BindingOptim
    ) -> BindingOptim:
        """Updates a BindingOptim with the specification's optimization steps.

        Args:
            binding_optim: The parent BindingOptim to be updated.

        Returns:
            The updated BindingOptim.
        """
        return binding_optim

    def update_input_length(
        self,
        left_shift: int = 0,
        right_shift: int = 0,
        min_len_shift: int = 0,
        max_len_shift: int = 0,
        new_min_len: int | None = None,
        new_max_len: int | None = None,
    ) -> None:
        """Updates input shapes, called by a child LayerSpec after its update.

        Args:
            left_shift: The change in size on the left side of the sequence.
            right_shift: The change in size on the right side of the sequence.
            min_len_shift: The change in the number of short input lengths.
            max_len_shift: The change in the number of long input lengths.
            new_min_len: The new `min_input_length`.
            new_max_len: The new `max_input_length`.
        """
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

    def lengths(self, seqs: Tensor) -> Tensor:
        r"""Counts the number of finite elements in each sequence.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{in_length})`.

        Returns:
            A tensor with the number of finite elements in each sequence of
            shape :math:`(\text{minibatch},)`.
        """
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


class ModeKey(tuple[LayerSpec], LengthManager):
    """Output of Mode.key() with length calculations."""

    @override
    def __str__(self) -> str:
        if all(i.name != "" for i in self):
            layer_str = "-".join(i.name for i in self)
            if "-" in layer_str:
                return f"{type(self).__name__}-[{layer_str}]"
            return f"{type(self).__name__}-{layer_str}"
        return self.__repr__()

    @override
    @property
    def out_channels(self) -> int:
        return self[-1].out_channels

    @override
    @property
    def in_channels(self) -> int:
        return self[0].in_channels

    @override
    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        for layer in iter(self):
            length = layer.out_len(length, mode=mode)
        return length

    @overload
    def in_len(self, length: T, mode: Literal["min"]) -> T: ...

    @overload
    def in_len(self, length: T, mode: Literal["max"]) -> T | None: ...

    @override
    def in_len(
        self, length: T, mode: Literal["min", "max"] = "max"
    ) -> T | None:
        for layer in reversed(self):
            out = layer.in_len(length, mode=mode)
            if out is None:
                return out
            length = out
        return length
