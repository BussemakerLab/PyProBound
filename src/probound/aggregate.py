"""Implementations of aggregate"""
from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from typing import Literal, cast

import torch
from torch import Tensor
from typing_extensions import Self, override

from . import __precision__
from .base import Binding, BindingOptim, Call, Component, Spec, Transform
from .containers import TModuleList


class Contribution(Transform):
    """Combines a _Binding with a log_alpha"""

    _unfreezable = Literal[Transform._unfreezable, "alpha"]

    def __init__(
        self,
        binding: Binding,
        train_alpha: bool = True,
        log_alpha: float = float("-inf"),
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.binding = binding
        self.train_alpha = train_alpha
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(log_alpha, dtype=__precision__),
            requires_grad=self.train_alpha,
        )

    @override
    def components(self) -> Iterator[Binding]:
        yield self.binding

    @override
    def unfreeze(self, parameter: _unfreezable = "all") -> None:
        """Unfreeze the desired parameter"""
        if parameter in ("alpha", "all") and self.train_alpha:
            if torch.isneginf(self.log_alpha):
                raise ValueError(
                    "Cannot unfreeze alpha without initializing it first"
                )
            self.log_alpha.requires_grad_()
        if parameter != "alpha":
            super().unfreeze(parameter)

    def expected_log_contribution(self) -> float:
        with torch.inference_mode():
            training = self.training
            self.eval()
            out = self(self.binding.expected_sequence())
            self.train(training)
            return cast(float, out.item())

    def set_contribution(
        self, log_contribution: float = 0, only_decrease: bool = False
    ) -> None:
        """Set alpha so E[self.forward] = log_contribution
        If only_decrease, only update if it would reduce its contribution"""
        log_alpha = log_contribution - self.binding.expected_log_score()
        if (
            torch.isneginf(self.log_alpha)
            or not only_decrease
            or log_alpha < self.log_alpha
        ):
            torch.nn.init.constant_(self.log_alpha, log_alpha)

    @override
    @Transform.cache
    def forward(self, seqs: Tensor) -> Tensor:
        return self.log_alpha + self.binding(seqs)


class Aggregate(Transform):
    """Collection of Contributions (corresponds to Partition in ProBound)"""

    _unfreezable = Literal[Transform._unfreezable, "concentration"]

    def __init__(
        self,
        contributions: Iterable[Contribution],
        train_concentration: bool = False,
        target_concentration: float = 1,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.train_concentration = train_concentration
        self.log_target_concentration: Tensor = torch.nn.Parameter(
            torch.tensor(math.log(target_concentration), dtype=__precision__),
            requires_grad=train_concentration,
        )

        # create contributions list
        self.contributions: TModuleList[Contribution] = TModuleList(
            contributions
        )
        binding = [ctrb.binding for ctrb in self.contributions]
        if len(binding) != len(set(binding)):
            raise ValueError(
                "All Binding components in an Aggregate must be unique"
            )

    @classmethod
    def from_binding(
        cls,
        binding: Iterable[Binding],
        train_concentration: bool = False,
        target_concentration: float = 1,
        name: str = "",
    ) -> Self:
        return cls(
            (Contribution(bmd) for bmd in binding),
            train_concentration=train_concentration,
            target_concentration=target_concentration,
            name=name,
        )

    @override
    def components(self) -> Iterator[Contribution]:
        return iter(self.contributions)

    @override
    def unfreeze(self, parameter: _unfreezable = "all") -> None:
        """Unfreeze the desired parameter"""
        if parameter in ("concentration", "all"):
            if self.train_concentration:
                self.log_target_concentration.requires_grad_()
        else:
            raise ValueError(
                f"{type(self).__name__} cannot unfreeze parameter {parameter}"
            )
        if parameter == "all":
            for cmpt in self._contributing():
                cmpt.unfreeze("all")

    @override
    def optim_procedure(
        self,
        ancestry: tuple[Component, ...] | None = None,
        current_order: dict[tuple[Spec, ...], BindingOptim] | None = None,
    ) -> dict[tuple[Spec, ...], BindingOptim]:
        # call "alpha_heuristic" during first step
        #     anywhere an aggregate is an ancestor
        current_order = super().optim_procedure(ancestry, current_order)

        key_to_ctrb = {ctrb.binding.key(): ctrb for ctrb in self.contributions}
        for key, binding_optim in current_order.items():
            ctrb = key_to_ctrb.get(key, None)
            if ctrb is not None:
                binding_optim.steps[0].calls.append(
                    Call(self, "alpha_heuristic", {"contribution": ctrb})
                )
                binding_optim.merge_binding_optim()

        return current_order

    def _contributing(self) -> Iterator[Contribution]:
        for ctrb in self.contributions:
            if not torch.isneginf(ctrb.log_alpha):
                yield ctrb

    def expected_log_aggregate(self) -> float:
        with torch.inference_mode():
            training = self.training
            self.eval()
            out = self(self.contributions[0].binding.expected_sequence())
            self.train(training)
            return cast(float, out.item())

    def alpha_heuristic(
        self, contribution: Contribution, frac: float = 0.05
    ) -> None:
        """Update a binding to contribute 5% of the expected aggregate"""
        if not 0 < frac < 1:
            raise ValueError(f"frac={frac} is not 0 < frac < 1")

        expected_log_aggregate = (
            self.expected_log_aggregate() - self.log_target_concentration
        )

        log_contribution = 0.0
        if expected_log_aggregate != float("-inf"):
            log_contribution = math.log(frac) + expected_log_aggregate

        only_decrease = False
        if not torch.isneginf(contribution.log_alpha):
            only_decrease = True

        contribution.set_contribution(
            log_contribution, only_decrease=only_decrease
        )

        if expected_log_aggregate != float("-inf"):
            log_ctrb_shift = math.log(
                2
                - math.exp(
                    self.expected_log_aggregate()
                    - expected_log_aggregate
                    - self.log_target_concentration
                )
            )
            for ctrb in self.contributions:
                if ctrb is contribution:
                    continue
                new_ctrb = log_ctrb_shift + ctrb.expected_log_contribution()
                if math.isfinite(new_ctrb):
                    ctrb.set_contribution(new_ctrb, only_decrease=True)

    @override
    @Transform.cache
    def forward(self, seqs: Tensor) -> Tensor:
        """Returns the normalized predicted aggregate (Z_{i} / [P])"""
        out = torch.full(
            (len(seqs),),
            float("-inf"),
            dtype=__precision__,
            device=self.contributions[0].log_alpha.device,
        )
        for ctrb in self._contributing():
            out = torch.logaddexp(out, ctrb(seqs))
        return self.log_target_concentration + out
