"""Definition of a binding mode as a series of layers applied sequentially.

Members are explicitly re-exported in pyprobound.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any, Literal, TypeVar, overload

import torch
from torch import Tensor
from typing_extensions import override

from . import __precision__
from .base import Binding, BindingOptim, Call, Component, Spec, Step, Transform
from .containers import TModuleList
from .layers import Layer, LengthManager, ModeKey
from .utils import clear_cache

logger = logging.getLogger(__name__)

T = TypeVar("T", int, Tensor)
Cooperativity = Any


class Mode(Binding, LengthManager):
    r"""Scores sequences with a series of layers applied sequentially.

    .. math::
        \frac{1}{K^{rel}_{\text{D}, a} (S_i)}
        = \sum_x \frac{1}{K^{rel}_{\text{D}, a} (S_{i, x})}

    Attributes:
        layers (TModuleList[Layer]): The layers to applied sequentially to an
            input sequence.
        log_hill (Tensor): The Hill coeffient in log space.
    """

    unfreezable = Literal[Binding.unfreezable, "hill"]

    def __init__(
        self, layers: Iterable[Layer], train_hill: bool = False, name: str = ""
    ) -> None:
        """Initializes the binding mode.

        Args:
            layers: The layers to be applied sequentially to an input sequence.
            train_hill: Whether to train a Hill coefficient.
            name: A string used to describe the binding mode.
        """
        super().__init__(name=name)
        self.layers: TModuleList[Layer] = TModuleList(layers)
        self._cooperativities: set["Cooperativity"] = set()
        for layer_idx, layer in enumerate(self.layers):
            layer._modes.add((self, layer_idx))

        if len(self.layers) == 0:
            raise ValueError(
                "Cannot create binding mode with empty layers argument"
            )

        # Store model attributes
        self.train_hill = train_hill
        self.log_hill = torch.nn.Parameter(
            torch.tensor(0, dtype=__precision__), requires_grad=train_hill
        )

        # Verify scoring model
        self.check_length_consistency()

    @override
    @property
    def out_channels(self) -> int:
        return self.key().out_channels

    @override
    @property
    def in_channels(self) -> int:
        return self.key().in_channels

    @property
    def input_shape(self) -> int:
        """The number of elements in an input sequence."""
        return self.layers[0].input_shape

    @property
    def min_input_length(self) -> int:
        """The minimum number of finite elements in an input sequence."""
        return self.layers[0].min_input_length

    @property
    def max_input_length(self) -> int:
        """The maximum number of finite elements in an input sequence."""
        return self.layers[0].max_input_length

    @override
    def key(self) -> ModeKey:
        return ModeKey(layer.layer_spec for layer in self.layers)

    @override
    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        return self.key().out_len(length=length, mode=mode)

    @overload
    def in_len(self, length: T, mode: Literal["min"]) -> T: ...

    @overload
    def in_len(self, length: T, mode: Literal["max"]) -> T | None: ...

    @override
    def in_len(
        self, length: T, mode: Literal["min", "max"] = "max"
    ) -> T | None:
        return self.key().in_len(length=length, mode=mode)

    @override
    def components(self) -> Iterator[Layer]:
        return iter(self.layers)

    @override
    def check_length_consistency(self) -> None:
        for layer in self.layers:
            layer.check_length_consistency()

        if len(self.layers) <= 1:
            return

        for layer_idx, (prev_layer, next_layer) in enumerate(
            zip(self.layers, self.layers[1:]), start=1
        ):
            if prev_layer.out_channels != next_layer.in_channels:
                raise RuntimeError(
                    f"expected {next_layer.in_channels} in_channels for"
                    f" layer {layer_idx}, found {prev_layer.out_channels}"
                )
            prev_layer_out_shape = prev_layer.out_len(
                prev_layer.input_shape, "shape"
            )
            if prev_layer_out_shape != next_layer.input_shape:
                raise RuntimeError(
                    f"expected input_shape {prev_layer_out_shape} for"
                    f" layer {layer_idx}, found {next_layer.input_shape}"
                )
            prev_layer_min_out_len = prev_layer.out_len(
                prev_layer.min_input_length, "min"
            )
            if prev_layer_min_out_len != next_layer.min_input_length:
                raise RuntimeError(
                    f"expected min_input_length {prev_layer_min_out_len} for"
                    f" layer {layer_idx}, found {next_layer.min_input_length}"
                )
            prev_layer_max_out_len = prev_layer.out_len(
                prev_layer.max_input_length, "max"
            )
            if prev_layer_max_out_len != next_layer.max_input_length:
                raise RuntimeError(
                    f"expected max_input_length {prev_layer_max_out_len} for"
                    f" layer {layer_idx}, found {next_layer.max_input_length}"
                )

    @override
    def unfreeze(self, parameter: unfreezable = "all") -> None:
        if parameter in ("hill", "all") and self.train_hill:
            self.log_hill.requires_grad_()
        if parameter != "hill":
            super().unfreeze(parameter)

    @override
    def optim_procedure(
        self,
        ancestry: tuple[Component, ...] | None = None,
        current_order: dict[tuple[Spec, ...], BindingOptim] | None = None,
    ) -> dict[tuple[Spec, ...], BindingOptim]:
        if ancestry is None:
            ancestry = tuple()
        if current_order is None:
            current_order = {}
        ancestry = ancestry + (self,)

        # Check if already in current_order
        if self.key() not in current_order:
            binding_optim = BindingOptim(
                {ancestry},
                (
                    [Step([Call(ancestry[0], "freeze", {})])]
                    if len(ancestry) > 0
                    else []
                ),
            )
            current_order[self.key()] = binding_optim
        else:
            binding_optim = current_order[self.key()]
            if ancestry in binding_optim.ancestry:
                return current_order
            binding_optim.ancestry.add(ancestry)

        # Unfreeze scoring parameters
        for layer in self.layers:
            layer.update_binding_optim(binding_optim)
        binding_optim.merge_binding_optim()

        # Unfreeze all parameters
        unfreeze_all = Step(
            [Call(ancestry[0], "unfreeze", {"parameter": "all"})]
        )
        if unfreeze_all not in binding_optim.steps:
            binding_optim.steps.append(unfreeze_all)

        return current_order

    @override
    def _apply_block(
        self, component: Component | None = None, cache_fun: str | None = None
    ) -> None:
        if component is not None and cache_fun is not None:
            logger.info(
                "Applying block of %s on %s.%s", component, self, cache_fun
            )
            self._blocking[cache_fun].add(component)
            logger.debug("%s._blocking=%s", self, self._blocking)

    @override
    def _release_block(
        self, component: Component | None = None, cache_fun: str | None = None
    ) -> None:
        if component is not None and cache_fun is not None:
            logger.info(
                "Releasing block of %s on %s.%s", component, self, cache_fun
            )
            self._blocking[cache_fun].discard(component)
            logger.debug("%s._blocking=%s", self, self._blocking)

            if len(self._blocking[cache_fun]) == 0:
                logger.info("Clearing cache of %s.%s", self, cache_fun)
                self._caches[cache_fun] = (None, None)
                logger.debug("%s._caches=%s", self, self._caches)
                clear_cache()

    def update_read_length(
        self,
        left_shift: int = 0,
        right_shift: int = 0,
        min_len_shift: int = 0,
        max_len_shift: int = 0,
        new_min_len: int | None = None,
        new_max_len: int | None = None,
    ) -> None:
        """Updates the input shape as part of a flank update.

        Args:
            left_shift: The change in size on the left side of the sequence.
            right_shift: The change in size on the right side of the sequence.
            min_len_shift: The change in the number of short input lengths.
            max_len_shift: The change in the number of long input lengths.
            new_min_len: The new `min_input_length`.
            new_max_len: The new `max_input_length`.
        """
        self._update_propagation(
            0,
            left_shift=left_shift,
            right_shift=right_shift,
            min_len_shift=min_len_shift,
            max_len_shift=max_len_shift,
            new_min_len=new_min_len,
            new_max_len=new_max_len,
        )

    def _update_propagation(
        self,
        layer_idx: int,
        left_shift: int = 0,
        right_shift: int = 0,
        min_len_shift: int = 0,
        max_len_shift: int = 0,
        new_min_len: int | None = None,
        new_max_len: int | None = None,
    ) -> None:
        """Updates input shapes, called by a child LayerSpec after its update.

        Args:
            layer_idx: The index of the layer.
            left_shift: The change in size on the left side of the sequence.
            right_shift: The change in size on the right side of the sequence.
            min_len_shift: The change in the number of short input lengths.
            max_len_shift: The change in the number of long input lengths.
            new_min_len: The new `min_input_length`.
            new_max_len: The new `max_input_length`.
        """
        if layer_idx >= len(self.layers):
            for coop in self._cooperativities:
                # pylint: disable-next=protected-access
                coop._update_propagation(
                    self,
                    left_shift=left_shift,
                    right_shift=right_shift,
                    min_len_shift=min_len_shift,
                    max_len_shift=max_len_shift,
                )
            return

        layer = self.layers[layer_idx]

        # Establish baseline
        old_shape = layer.out_len(layer.input_shape)
        old_min_len = layer.out_len(layer.min_input_length, mode="min")
        old_max_len = layer.out_len(layer.max_input_length, mode="max")

        # Get shifts
        left_shape = layer.out_len(layer.input_shape + left_shift)
        right_shape = layer.out_len(
            layer.input_shape + right_shift + left_shift
        )

        # Apply update
        layer.update_input_length(
            left_shift=left_shift,
            right_shift=right_shift,
            min_len_shift=min_len_shift,
            max_len_shift=max_len_shift,
            new_min_len=new_min_len,
            new_max_len=new_max_len,
        )

        # Get new lengths
        new_min_len = layer.out_len(layer.min_input_length, mode="min")
        new_max_len = layer.out_len(layer.max_input_length, mode="max")

        # Propagate update
        layer.check_length_consistency()
        self._update_propagation(
            layer_idx + 1,
            left_shift=left_shape - old_shape,
            right_shift=right_shape - left_shape,
            min_len_shift=new_min_len - old_min_len - right_shape + old_shape,
            max_len_shift=new_max_len - old_max_len - right_shape + old_shape,
            new_min_len=new_min_len,
            new_max_len=new_max_len,
        )

    @override
    def expected_sequence(self) -> Tensor:
        return torch.full(
            size=(1, self.in_channels, self.input_shape),
            fill_value=1 / self.in_channels,
            dtype=__precision__,
            device=self.log_hill.device,
        )

    @override
    @Transform.cache
    def score_windows(self, seqs: Tensor) -> Tensor:
        r"""Calculates the log score of each window before summing over them.

        .. math::
            \log \frac{1}{K^{rel}_{\text{D}, a} (S_{i, x})}

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{in_length})`.

        Returns:
            A tensor with the log score of each window of shape
            :math:`(\text{minibatch},\text{out_channels},\text{out_length})`.
        """
        for module in self.layers:
            seqs = module(seqs)
        return seqs

    @override
    @Transform.cache
    def forward(self, seqs: Tensor) -> Tensor:
        r"""Calculates the log score of each sequence.

        .. math::
            \log \frac{1}{K^{rel}_{\text{D}, a} (S_i)}
            = \log \sum_x \frac{1}{K^{rel}_{\text{D}, a} (S_{i, x})}

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            The log score tensor of shape :math:`(\text{minibatch},)`.
        """
        return (torch.exp(self.log_hill) * self.score_windows(seqs)).logsumexp(
            (1, 2)
        )
