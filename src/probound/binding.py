"""Binding and selection models"""

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
from .layers import BindingModeKey, Layer
from .utils import clear_cache

logger = logging.getLogger(__name__)

T = TypeVar("T", int, Tensor)
BindingCooperativity = Any


class BindingMode(Binding):
    """Scores probes using weight matrices for each interaction distance"""

    _unfreezable = Literal[Binding._unfreezable, "hill"]

    def __init__(
        self, layers: Iterable[Layer], train_hill: bool = False, name: str = ""
    ) -> None:
        super().__init__(name=name)
        self.layers = TModuleList(layers)
        self._binding_cooperativity: set["BindingCooperativity"] = set()
        for layer_idx, layer in enumerate(self.layers):
            layer._binding_mode.add((self, layer_idx))

        # store model attributes
        self.train_hill = train_hill
        self.hill = torch.nn.Parameter(
            torch.tensor(0, dtype=__precision__), requires_grad=train_hill
        )

        # verify scoring model
        self.check_length_consistency()

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

    @override
    def components(self) -> Iterator[Layer]:
        return iter(self.layers)

    @override
    def unfreeze(self, parameter: _unfreezable = "all") -> None:
        """Unfreeze the desired parameter"""
        if parameter in ("hill", "all") and self.train_hill:
            self.hill.requires_grad_()
        if parameter != "hill":
            super().unfreeze(parameter)

    @property
    def out_channels(self) -> int:
        return self.layers[-1].out_channels

    @property
    def in_channels(self) -> int:
        return self.layers[0].in_channels

    @property
    def input_shape(self) -> int:
        return self.layers[0].input_shape

    @property
    def min_input_length(self) -> int:
        return self.layers[0].min_input_length

    @property
    def max_input_length(self) -> int:
        return self.layers[0].max_input_length

    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        """Output length"""
        return self.key().out_len(length=length, mode=mode)

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
        return self.key().in_len(length=length, mode=mode)

    def update_read_length(
        self,
        left_shift: int = 0,
        right_shift: int = 0,
        min_len_shift: int = 0,
        max_len_shift: int = 0,
        new_min_len: int | None = None,
        new_max_len: int | None = None,
    ) -> None:
        """Update parameters after shifting input flank lengths"""
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
        if layer_idx >= len(self.layers):
            for coop in self._binding_cooperativity:
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

        # establish baseline
        old_shape = layer.out_len(layer.input_shape)
        old_min_len = layer.out_len(layer.min_input_length, mode="min")
        old_max_len = layer.out_len(layer.max_input_length, mode="max")

        # get shifts
        left_shape = layer.out_len(layer.input_shape + left_shift)
        right_shape = layer.out_len(
            layer.input_shape + right_shift + left_shift
        )

        # apply update
        layer.update_input_length(
            left_shift=left_shift,
            right_shift=right_shift,
            min_len_shift=min_len_shift,
            max_len_shift=max_len_shift,
            new_min_len=new_min_len,
            new_max_len=new_max_len,
        )

        # get new lengths
        new_min_len = layer.out_len(layer.min_input_length, mode="min")
        new_max_len = layer.out_len(layer.max_input_length, mode="max")

        # propagate update
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
    def key(self) -> BindingModeKey:
        return BindingModeKey(layer.layer_spec for layer in self.layers)

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

        # check if already in current_order
        if self.key() not in current_order:
            binding_optim = BindingOptim(
                {ancestry},
                [Step([Call(ancestry[0], "freeze", {})])]
                if len(ancestry) > 0
                else [],
            )
            current_order[self.key()] = binding_optim
        else:
            binding_optim = current_order[self.key()]
            if ancestry in binding_optim.ancestry:
                return current_order
            binding_optim.ancestry.add(ancestry)

        # unfreeze hill
        if self.train_hill:
            binding_optim.steps.append(
                Step([Call(self, "unfreeze", {"parameter": "hill"})])
            )

        # unfreeze scoring parameters
        for layer in self.layers:
            layer.update_binding_optim(binding_optim)
        binding_optim.merge_binding_optim()

        # unfreeze all parameters
        unfreeze_all = Step(
            [Call(ancestry[0], "unfreeze", {"parameter": "all"})]
        )
        if unfreeze_all not in binding_optim.steps:
            binding_optim.steps.append(unfreeze_all)

        return current_order

    @override
    def expected_sequence(self) -> Tensor:
        """Uninformative prior of a sequence"""
        return torch.full(
            size=(1, self.in_channels, self.input_shape),
            fill_value=1 / self.in_channels,
            dtype=__precision__,
            device=self.hill.device,
        )

    @override
    @Transform.cache
    def score_windows(self, seqs: Tensor) -> Tensor:
        for module in self.layers:
            seqs = module(seqs)
        return seqs

    @override
    @Transform.cache
    def forward(self, seqs: Tensor) -> Tensor:
        return (torch.exp(self.hill) * self.score_windows(seqs)).logsumexp(
            (1, 2)
        )
