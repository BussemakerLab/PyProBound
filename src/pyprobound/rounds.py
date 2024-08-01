"""Selection layers corresponding to sequencing rounds / CountTable columns."""

from __future__ import annotations

import abc
from collections.abc import Iterable, Iterator
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self, override

from . import __precision__
from .aggregate import Aggregate
from .base import Binding, BindingOptim, Call, Component, Spec, Transform
from .utils import log1mexp


def repeat_round(
    binding: Iterable[Binding], n_rounds: int, round_type: type[Round]
) -> list[InitialRound | Round]:
    """Creates multiple sequential enrichment rounds from binding components.

    Args:
        binding: The binding components driving enrichment.
        n_rounds: The total number of sequentially enriched libraries.
        round_type: The sequencing round type to be repeated.

    Returns:
        A list of `n_rounds` rounds.
    """
    rounds: list[InitialRound | Round] = []
    for round_idx in range(n_rounds):
        if round_idx == 0:
            rounds.append(InitialRound())
        else:
            rounds.append(round_type.from_binding(binding, rounds[-1]))
    return rounds


class BaseRound(Transform, abc.ABC):
    r"""Base class for sequencing rounds.

    Attributes:
        reference_round (BaseRound | None): The previous round for cumulative
            enrichment.
        log_depth (Tensor): The sequencing depth :math:`\eta` in log space.
    """

    unfreezable = Literal[Transform.unfreezable, "depth"]

    def __init__(
        self,
        reference_round: BaseRound | None,
        train_depth: bool = True,
        log_depth: float = 0,
        library_concentration: float = -1,
        name: str = "",
    ) -> None:
        r"""Initializes the round.

        Args:
            reference_round: The previous round used in cumulative enrichment.
            train_depth: Whether to train the sequencing depth :math:`\eta`.
            log_depth: The initial value of `log_depth`.
            library_concentration: The total library concentration, used for
                estimating the free protein concentration.
            name: A string used to describe the round.
        """
        super().__init__(name=name)
        self.reference_round = reference_round
        self.train_depth = train_depth
        self.log_depth = torch.nn.Parameter(
            torch.tensor(log_depth, dtype=__precision__),
            requires_grad=train_depth,
        )
        self._library_concentration = library_concentration

    @property
    def library_concentration(self) -> float:
        r"""The total library concentration :math:`[\text{library}]`."""
        if self._library_concentration < 0:
            raise ValueError(
                f"{self} not initialized with 'library_concentration'"
            )
        return self._library_concentration

    @override
    def __setattr__(self, name: str, value: Any) -> None:
        # Avoid registering reference_round as a submodule
        if name == "reference_round":
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    @override
    @abc.abstractmethod
    def components(self) -> Iterator[Aggregate]: ...

    @override
    def unfreeze(self, parameter: unfreezable = "all") -> None:
        if parameter in ("depth", "all") and self.train_depth:
            self.log_depth.requires_grad_()
        if parameter != "depth":
            super().unfreeze(parameter)

    @override
    def optim_procedure(
        self,
        ancestry: tuple[Component, ...] | None = None,
        current_order: dict[tuple[Spec, ...], BindingOptim] | None = None,
    ) -> dict[tuple[Spec, ...], BindingOptim]:
        # Unfreeze eta during first step anywhere a round is an ancestor
        new_current_order = super().optim_procedure(ancestry, current_order)

        for key, binding_optim in new_current_order.items():
            if any(self in ancestors for ancestors in binding_optim.ancestry):
                new_current_order[key].steps[0].calls.append(
                    Call(self, "unfreeze", {"parameter": "depth"})
                )

        return new_current_order

    @override
    def _apply_block(
        self, component: Component | None = None, cache_fun: str | None = None
    ) -> None:
        super()._apply_block(component, cache_fun)
        if self.reference_round is not None:
            # pylint: disable-next=protected-access
            self.reference_round._apply_block(
                self, "log_cumulative_enrichment"
            )

    @override
    def _release_block(
        self, component: Component | None = None, cache_fun: str | None = None
    ) -> None:
        super()._release_block(component, cache_fun)
        if self.reference_round is not None:
            # pylint: disable-next=protected-access
            self.reference_round._release_block(
                self, "log_cumulative_enrichment"
            )

    @abc.abstractmethod
    def log_aggregate(self, seqs: Tensor) -> Tensor:
        r"""Predicts the log aggregate :math:`\log Z_i`.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            The log aggregate tensor of shape :math:`(\text{minibatch},)`.
        """

    @abc.abstractmethod
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        r"""Predicts the log enrichment ratio.

        .. math::
            \log \frac{f_{i,r}}{f_{i,r-1}}

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            The log enrichment ratio tensor of shape
            :math:`(\text{minibatch},)`.
        """

    @Transform.cache
    def log_cumulative_enrichment(self, seqs: Tensor) -> Tensor:
        r"""Predicts the log cumulative enrichment.

        .. math::
            \log \prod_r f_{i,r}

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            The log enrichment ratio tensor of shape
            :math:`(\text{minibatch},)`.
        """
        if (
            isinstance(self.reference_round, InitialRound)
            or self.reference_round is None
        ):
            return self.log_enrichment(seqs)
        return self.reference_round.log_cumulative_enrichment(
            seqs
        ) + self.log_enrichment(seqs)

    @override
    @Transform.cache
    def forward(self, seqs: Tensor) -> Tensor:
        r"""Predicts the log relative count.

        .. math::
            \log \eta_{i,r} \prod_r f_{i,r}

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            The log relative count tensor of shape :math:`(\text{minibatch},)`.
        """
        return self.log_depth + self.log_cumulative_enrichment(seqs)


class InitialRound(BaseRound):
    """Initial sequenced round, outputs 1 by convention."""

    def __init__(self, name: str = "") -> None:
        r"""Initializes the round.

        Args:
            name: A string used to describe the round.
        """
        super().__init__(reference_round=None, train_depth=False, name=name)

    @override
    def components(self) -> Iterator[Aggregate]:
        return iter(())

    @override
    def log_aggregate(self, seqs: Tensor) -> Tensor:
        return self.log_enrichment(seqs)

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return torch.zeros(
            len(seqs), dtype=__precision__, device=self.log_depth.device
        )


class Round(BaseRound):
    r"""Base class for sequenced rounds with an aggregate object.

    Attributes:
        aggregate (Aggregate): The binding components driving enrichment.
        reference_round (BaseRound): The previous round for cumulative
            enrichment.
        log_depth (Tensor): The sequencing depth :math:`\eta` in log space.
    """

    def __init__(
        self,
        aggregate: Aggregate,
        reference_round: BaseRound | None,
        train_depth: bool = True,
        log_depth: float = 0,
        library_concentration: float = -1,
        name: str = "",
    ) -> None:
        r"""Initializes the round.

        Args:
            aggregate: The binding components driving enrichment.
            reference_round: The previous round used in cumulative enrichment.
            train_depth: Whether to train the sequencing depth :math:`\eta`.
            log_depth: The initial value of `log_depth`.
            library_concentration: The total library concentration, used for
                estimating the free protein concentration.
            name: A string used to describe the round.
        """
        super().__init__(
            reference_round,
            train_depth=train_depth,
            log_depth=log_depth,
            library_concentration=library_concentration,
            name=name,
        )
        self.aggregate = aggregate
        self.reference_round: BaseRound | None

    @classmethod
    def from_binding(
        cls,
        binding: Iterable[Binding],
        reference_round: BaseRound | None,
        train_depth: bool = True,
        log_depth: float = 0,
        train_concentration: bool = False,
        target_concentration: float = 1,
        library_concentration: float = -1,
        activity_heuristic: float = 0.05,
        name: str = "",
    ) -> Self:
        r"""Creates a new instance from binding components and reference round.

        Args:
            binding: The binding components driving enrichment.
            reference_round: The previous round used in cumulative enrichment.
            train_depth: Whether to train the sequencing depth :math:`\eta`.
            log_depth: The initial value of `log_depth`.
            train_concentration: Whether to train the protein concentration.
            target_concentration: Protein concentration, used for estimating
                the free protein concentration.
            library_concentration: Library concentration, used for estimating
                the free protein concentration.
            activity_heuristic: The fraction of the total aggregate that the
                contribution will be set to when it is first optimized.
            name: A string used to describe the round.
        """
        return cls(
            Aggregate.from_binding(
                binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
                activity_heuristic=activity_heuristic,
            ),
            reference_round=reference_round,
            train_depth=train_depth,
            log_depth=log_depth,
            library_concentration=library_concentration,
            name=name,
        )

    @override
    def components(self) -> Iterator[Aggregate]:
        yield self.aggregate

    @override
    def log_aggregate(self, seqs: Tensor) -> Tensor:
        return self.aggregate(seqs)


class BoundRound(Round):
    r"""Saturated enrichment round.

    .. math::
        \frac{f_{i,r}}{f_{i,r-1}} = \frac{Z_{i,r}}{1 + Z_{i,r}}

    Attributes:
        aggregate (Aggregate): The binding components driving enrichment.
        reference_round (BaseRound): The previous round for cumulative
            enrichment.
        log_depth (Tensor): The sequencing depth :math:`\eta` in log space.
    """

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return F.logsigmoid(self.log_aggregate(seqs))


class BoundUnsaturatedRound(Round):
    r"""Unsaturated enrichment round.

    .. math::
        \frac{f_{i,r}}{f_{i,r-1}} = Z_{i,r}

    Attributes:
        aggregate (Aggregate): The binding components driving enrichment.
        reference_round (BaseRound): The previous round for cumulative
            enrichment.
        log_depth (Tensor): The sequencing depth :math:`\eta` in log space.
    """

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return self.log_aggregate(seqs)


class UnboundRound(Round):
    r"""Free fraction enrichment round.

    .. math::
        \frac{f_{i,r}}{f_{i,r-1}} = \frac{1}{1 + Z_{i,r}}

    Attributes:
        aggregate (Aggregate): The binding components driving enrichment.
        reference_round (BaseRound): The previous round for cumulative
            enrichment.
        log_depth (Tensor): The sequencing depth :math:`\eta` in log space.
    """

    @classmethod
    def from_round(
        cls, base_round: Round, train_depth: bool = True, name: str = ""
    ) -> Self:
        r"""Creates a new instance from the round modeling the bound fraction.

        The resulting round has the same `aggregate` and `reference_round`
        as `base_round`.

        Args:
            base_round: The round modeling the bound fraction.
            train_depth: Whether to train the sequencing depth :math:`\eta`.
            name: A string used to describe the round.
        """
        return cls(
            base_round.aggregate,
            base_round.reference_round,
            train_depth=train_depth,
            # pylint: disable-next=protected-access
            library_concentration=base_round._library_concentration,
            name=name,
        )

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return F.logsigmoid(-self.log_aggregate(seqs))


class RhoGammaRound(Round):
    r"""Enrichment round with learned saturation parameters.

    .. math::
        \frac{f_{i,r}}{f_{i,r-1}} = \frac{Z_{i,r}^\rho}{(1+Z_{i,r})^\gamma}

    Attributes:
        aggregate (Aggregate): The binding components driving enrichment.
        reference_round (BaseRound): The previous round for cumulative
            enrichment.
        log_depth (Tensor): The sequencing depth :math:`\eta` in log space.
        rho (Tensor): The parameter :math:`\rho`.
        gamma (Tensor): The parameter :math:`\gamma`.
    """

    unfreezable = Literal[Round.unfreezable, "rho"]

    def __init__(
        self,
        aggregate: Aggregate,
        reference_round: BaseRound | None,
        train_depth: bool = True,
        log_depth: float = 0,
        library_concentration: float = -1,
        rho: float = 0,
        gamma: float = -1,
        name: str = "",
    ) -> None:
        r"""Initializes the round.

        Args:
            aggregate: The binding components driving enrichment.
            reference_round: The previous round used in cumulative enrichment.
            train_depth: Whether to train the sequencing depth :math:`\eta`.
            log_depth: The initial value of `log_depth`.
            library_concentration: The total library concentration, used for
                estimating the free protein concentration.
            rho: The initial value of `rho`.
            gamma: The initial value of `gamma`.
            name: A string used to describe the round.
        """
        super().__init__(
            aggregate,
            reference_round,
            train_depth=train_depth,
            log_depth=log_depth,
            library_concentration=library_concentration,
            name=name,
        )
        self.rho = torch.nn.Parameter(torch.tensor(rho, dtype=__precision__))
        self.gamma = torch.nn.Parameter(
            torch.tensor(gamma, dtype=__precision__)
        )

    @override
    @classmethod
    def from_binding(
        cls,
        binding: Iterable[Binding],
        reference_round: BaseRound | None,
        train_depth: bool = True,
        log_depth: float = 0,
        train_concentration: bool = False,
        target_concentration: float = 1,
        library_concentration: float = -1,
        activity_heuristic: float = 0.05,
        name: str = "",
        rho: float = 1,
        gamma: float = -1,
    ) -> Self:
        r"""Creates a new instance from binding components and reference round.

        Args:
            binding: The binding components driving enrichment.
            reference_round: The previous round used in cumulative enrichment.
            train_depth: Whether to train the sequencing depth :math:`\eta`.
            log_depth: The initial value of `log_depth`.
            train_concentration: Whether to train the protein concentration.
            target_concentration: Protein concentration, used for estimating
                the free protein concentration.
            library_concentration: Library concentration, used for estimating
                the free protein concentration.
            activity_heuristic: The fraction of the total aggregate that the
                contribution will be set to when it is first optimized.
            name: A string used to describe the round.
            rho: The initial value of `rho`.
            gamma: The initial value of `gamma`.
        """
        return cls(
            Aggregate.from_binding(
                binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
                activity_heuristic=activity_heuristic,
            ),
            reference_round=reference_round,
            train_depth=train_depth,
            log_depth=log_depth,
            library_concentration=library_concentration,
            rho=rho,
            gamma=gamma,
            name=name,
        )

    @override
    def unfreeze(self, parameter: unfreezable = "all") -> None:
        if parameter in ("rho", "all"):
            self.rho.requires_grad_()
            self.gamma.requires_grad_()
        if parameter != "rho":
            super().unfreeze(parameter)

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        log_agg = self.log_aggregate(seqs)
        return self.rho * log_agg - self.gamma * F.logsigmoid(-log_agg)


class ExponentialRound(Round):
    r"""Exponential enrichment round used for modeling catalytic experiments.

    .. math::
        \frac{f_{i,r}}{f_{i,r-1}} = \sigma(\delta) \times e^{-Z_{i,r}}
            + \sigma(-\delta) \times \left( 1 - e^{-Z_{i,r}} \right)

    Attributes:
        aggregate (Aggregate): The binding components driving enrichment.
        reference_round (BaseRound): The previous round for cumulative
            enrichment.
        log_depth (Tensor): The sequencing depth :math:`\eta` in log space.
        log_delta (Tensor): The parameter :math:`\log\delta` in log space.
    """

    unfreezable = Literal[Round.unfreezable, "delta"]

    def __init__(
        self,
        aggregate: Aggregate,
        reference_round: BaseRound | None,
        train_depth: bool = True,
        log_depth: float = 0,
        library_concentration: float = -1,
        delta: float = -15,
        train_delta: bool = True,
        name: str = "",
    ) -> None:
        r"""Initializes the round.

        Args:
            aggregate: The binding components driving enrichment.
            reference_round: The previous round used in cumulative enrichment.
            train_depth: Whether to train the sequencing depth :math:`\eta`.
            log_depth: The initial value of `log_depth`.
            library_concentration: The total library concentration, used for
                estimating the free protein concentration.
            delta: The initial value of :math:`\delta`.
            train_delta: Whether to train the `delta` parameter.
            name: A string used to describe the round.
        """
        super().__init__(
            aggregate,
            reference_round,
            train_depth=train_depth,
            log_depth=log_depth,
            library_concentration=library_concentration,
            name=name,
        )
        self.delta = torch.nn.Parameter(
            torch.tensor(delta, dtype=__precision__)
        )
        self.train_delta = train_delta

    @classmethod
    def from_round(
        cls,
        base_round: Self,
        reference_round: BaseRound | None = None,
        train_depth: bool = True,
        target_concentration: float = 1,
        name: str = "",
    ) -> Self:
        r"""Creates a new instance from another round at a different timepoint.

        The resulting round has the same `aggregate` and `delta` as
        `base_round`.

        Args:
            base_round: The binding components driving enrichment.
            reference_round: The previous round used in cumulative enrichment.
            train_depth: Whether to train the sequencing depth :math:`\eta`.
            name: A string used to describe the round.
        """
        if reference_round is None:
            reference_round = base_round.reference_round
        return cls(
            Aggregate(
                base_round.aggregate.contributions,
                train_concentration=base_round.aggregate.train_concentration,
                target_concentration=target_concentration,
            ),
            reference_round,
            train_depth=train_depth,
            delta=base_round.delta.item(),
            train_delta=base_round.train_delta,
            name=name,
        )

    @override
    @classmethod
    def from_binding(
        cls,
        binding: Iterable[Binding],
        reference_round: BaseRound | None,
        train_depth: bool = True,
        log_depth: float = 0,
        train_concentration: bool = False,
        target_concentration: float = 1,
        library_concentration: float = -1,
        activity_heuristic: float = 0.05,
        name: str = "",
        delta: float = -15,
        train_delta: bool = True,
    ) -> Self:
        r"""Creates a new instance from binding components and reference round.

        Args:
            binding: The binding components driving enrichment.
            reference_round: The previous round used in cumulative enrichment.
            train_depth: Whether to train the sequencing depth :math:`\eta`.
            log_depth: The initial value of `log_depth`.
            train_concentration: Whether to train the protein concentration.
            target_concentration: Protein concentration, used for estimating
                the free protein concentration.
            library_concentration: Library concentration, used for estimating
                the free protein concentration.
            activity_heuristic: The fraction of the total aggregate that the
                contribution will be set to when it is first optimized.
            name: A string used to describe the round.
            delta: The initial value of :math:`\delta`.
            train_delta: Whether to train the `delta` parameter.
        """
        return cls(
            Aggregate.from_binding(
                binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
                activity_heuristic=activity_heuristic,
            ),
            reference_round=reference_round,
            train_depth=train_depth,
            log_depth=log_depth,
            library_concentration=library_concentration,
            delta=delta,
            train_delta=train_delta,
            name=name,
        )

    @override
    def unfreeze(self, parameter: unfreezable = "all") -> None:
        if parameter in ("delta", "all") and self.train_delta:
            self.delta.requires_grad_()
        if parameter != "delta":
            super().unfreeze(parameter)

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        aggregate = torch.exp(self.log_aggregate(seqs))
        if not torch.isneginf(self.delta):
            return torch.logaddexp(
                F.logsigmoid(self.delta) - aggregate,
                F.logsigmoid(-self.delta) + log1mexp(aggregate),
            )
        return log1mexp(aggregate)
