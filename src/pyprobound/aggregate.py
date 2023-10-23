"""Weighted sum over an aggregate of different binding modes.

Members are explicitly re-exported in pyprobound.
"""
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
    r"""Models the activity :math:`\alpha` of a Binding component.

    .. math::
        \alpha_a = \frac{[\text{P}_a]}{K_{\text{D}, a} (S_0)}

    Attributes:
        binding (Binding): The binding component whose contribution is modeled.
        log_activity (Tensor): The activity :math:`\alpha` in log space.
    """

    unfreezable = Literal[Transform.unfreezable, "activity"]

    def __init__(
        self,
        binding: Binding,
        train_activity: bool = True,
        log_activity: float = float("-inf"),
        activity_heuristic: float = 0.05,
        name: str = "",
    ) -> None:
        r"""Initializes the contribution model.

        Args:
            binding: The Binding component whose contribution will be modeled.
            train_activity: Whether to train the activity :math:`\alpha`.
            log_activity: The initial value of `log_activity`.
            activity_heuristic: The fraction of the total aggregate that the
                contribution will be set to when it is first optimized.
            name: A string used to describe the round.
        """
        super().__init__(name=name)

        if not 0 < activity_heuristic < 1:
            raise ValueError(
                f"activity_heuristic={activity_heuristic} is not 0 < frac < 1"
            )

        self.binding = binding
        self.train_activity = train_activity
        self.log_activity = torch.nn.Parameter(
            torch.tensor(log_activity, dtype=__precision__),
            requires_grad=self.train_activity,
        )
        self.activity_heuristic = activity_heuristic

    @override
    def components(self) -> Iterator[Binding]:
        yield self.binding

    @override
    def unfreeze(self, parameter: unfreezable = "all") -> None:
        if parameter in ("activity", "all") and self.train_activity:
            if torch.isneginf(self.log_activity):
                raise ValueError(
                    "Cannot unfreeze activity without initializing it first"
                )
            self.log_activity.requires_grad_()
        if parameter != "activity":
            super().unfreeze(parameter)

    def expected_log_contribution(self) -> float:
        r"""Calculates the expected log contribution.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            The expected log contribution represented as a float.
        """
        with torch.inference_mode():
            training = self.training
            self.eval()
            out = self(self.binding.expected_sequence())
            self.train(training)
            return cast(float, out.item())

    @override
    @Transform.cache
    def forward(self, seqs: Tensor) -> Tensor:
        r"""Calculates the log contribution of a Binding to the Aggregate.

        .. math::
            \log \frac{\alpha_a}{K^{rel}_{\text{D}, a} (S_i)}

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            The log contribution tensor of shape :math:`(\text{minibatch},)`.
        """
        return self.log_activity + self.binding(seqs)

    def set_contribution(
        self, log_contribution: float = 0, only_decrease: bool = False
    ) -> None:
        """Sets the activity so that E[log contribution] = `log_contribution`.

        Args:
            log_contribution: The new value of the expected log contribution.
            only_decrease: If True, update the activity only if the
                contribution would decrease
        """
        log_activity = log_contribution - self.binding.expected_log_score()
        if (
            torch.isneginf(self.log_activity)
            or not only_decrease
            or log_activity < self.log_activity
        ):
            torch.nn.init.constant_(self.log_activity, log_activity)


class Aggregate(Transform):
    r"""Models the weighted sum over an aggregate of different binding modes.

    .. math::
        Z_{i} = \sum_a \frac{\alpha_a}{K^{rel}_{\text{D}, a} (S_i)}

    Attributes:
        contributions (TModuleList[Contribution]): The Contributions making up
            the aggregate.
        log_target_concentration (Tensor): The total protein concentration
            :math:`[\text{P}]_T` in log space.
    """
    unfreezable = Literal[Transform.unfreezable, "concentration"]

    def __init__(
        self,
        contributions: Iterable[Contribution],
        train_concentration: bool = False,
        target_concentration: float = 1,
        name: str = "",
    ) -> None:
        """Initializes the aggregate.

        Args:
            contributions: The contributions making up the aggregate.
            train_concentration: Whether to train the protein concentration.
            target_concentration: Protein concentration, used for estimating
                the free protein concentration.
            name: A string used to describe the aggregate.
        """
        super().__init__(name=name)
        self.train_concentration = train_concentration
        self.log_target_concentration: Tensor = torch.nn.Parameter(
            torch.tensor(math.log(target_concentration), dtype=__precision__),
            requires_grad=train_concentration,
        )

        # Create contributions list
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
        activity_heuristic: float = 0.05,
        name: str = "",
    ) -> Self:
        r"""Creates a new instance from the binding modes.

        Args:
            binding: The binding modes driving enrichment.
            train_concentration: Whether to train the protein concentration.
            target_concentration: Protein concentration, used for estimating
                the free protein concentration.
            activity_heuristic: The fraction of the total aggregate that the
                contribution will be set to when it is first optimized.
            name: A string used to describe the round.
        """
        return cls(
            (
                Contribution(bmd, activity_heuristic=activity_heuristic)
                for bmd in binding
            ),
            train_concentration=train_concentration,
            target_concentration=target_concentration,
            name=name,
        )

    @override
    def components(self) -> Iterator[Contribution]:
        return iter(self.contributions)

    def _contributing(self) -> Iterator[Contribution]:
        """Iterator over Contributions that contribute to the aggregate."""
        for ctrb in self.contributions:
            if not torch.isneginf(ctrb.log_activity):
                yield ctrb

    @override
    def unfreeze(self, parameter: unfreezable = "all") -> None:
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
        # Call "activity_heuristic" during first step
        #     anywhere an aggregate is an ancestor
        current_order = super().optim_procedure(ancestry, current_order)

        key_to_ctrb = {ctrb.binding.key(): ctrb for ctrb in self.contributions}
        for key, binding_optim in current_order.items():
            ctrb = key_to_ctrb.get(key, None)
            if ctrb is not None:
                binding_optim.steps[0].calls.append(
                    Call(self, "activity_heuristic", {"contribution": ctrb})
                )
                binding_optim.merge_binding_optim()

        return current_order

    def expected_log_aggregate(self) -> float:
        r"""Calculates the expected log aggregate.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            The expected log aggregate represented as a float.
        """
        with torch.inference_mode():
            training = self.training
            self.eval()
            out = self(self.contributions[0].binding.expected_sequence())
            self.train(training)
            return cast(float, out.item())

    @override
    @Transform.cache
    def forward(self, seqs: Tensor) -> Tensor:
        r"""Calculates the log aggregate :math:`\log Z_i`.

        .. math::
            \log Z_{i} = \log\sum_a\frac{\alpha_a}{K^{rel}_{\text{D}, a} (S_i)}

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            The log aggregate tensor of shape :math:`(\text{minibatch},)`.
        """
        out = torch.full(
            (len(seqs),),
            float("-inf"),
            dtype=__precision__,
            device=self.contributions[0].log_activity.device,
        )
        for ctrb in self._contributing():
            out = torch.logaddexp(out, ctrb(seqs))
        return self.log_target_concentration + out

    def activity_heuristic(
        self, contribution: Contribution, frac: float | None = None
    ) -> None:
        """Sets the activity so that E[contribution] = frac * E[aggregate].

        Args:
            contribution: The Contribution whose activity will be updated.
            frac: The proportion of the aggregate the Binding contributes. If
                None, uses the activity_heuristic attribute of Contribution.
        """
        if frac is None:
            frac = contribution.activity_heuristic
        if not 0 < frac < 1:
            raise ValueError(f"frac={frac} is not 0 < frac < 1")

        expected_log_aggregate = (
            self.expected_log_aggregate() - self.log_target_concentration
        )

        log_contribution = 0.0
        if expected_log_aggregate != float("-inf"):
            log_contribution = math.log(frac) + expected_log_aggregate

        only_decrease = False
        if not torch.isneginf(contribution.log_activity):
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
