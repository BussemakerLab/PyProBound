"""Implementations of different rounds"""
from __future__ import annotations

import abc
from collections.abc import Iterable, Iterator
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self, override

from . import __precision__
from .aggregate import Aggregate
from .base import Binding, BindingOptim, Call, Component, Spec, Step, Transform
from .containers import Buffer
from .utils import log1mexp, logsigmoid


class _ARound(Transform, abc.ABC):
    """Modeling a sequencing round"""

    _unfreezable = Literal[Transform._unfreezable, "eta"]

    def __init__(
        self,
        reference_round: _ARound | None,
        train_eta: bool = True,
        log_eta: float = 0,
        library_concentration: float = -1,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.reference_round = reference_round
        self.train_eta = train_eta
        self.log_eta = torch.nn.Parameter(
            torch.tensor(log_eta, dtype=__precision__), requires_grad=train_eta
        )
        self._library_concentration: Tensor = Buffer(
            torch.tensor(library_concentration, dtype=__precision__)
        )

    @property
    def library_concentration(self) -> Tensor:
        if self._library_concentration < 0:
            raise ValueError(
                f"{self} not initialized with 'library_concentration'"
            )
        return self._library_concentration

    @override
    def __setattr__(self, name: str, value: Any) -> None:
        # avoid registering reference_round as a submodule
        if name == "reference_round":
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

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

    @override
    @abc.abstractmethod
    def components(self) -> Iterator[Aggregate]:
        ...

    @override
    def unfreeze(self, parameter: _unfreezable = "all") -> None:
        """Unfreeze the desired parameter"""
        if parameter in ("eta", "all") and self.train_eta:
            self.log_eta.requires_grad_()
        if parameter != "eta":
            super().unfreeze(parameter)

    @override
    def optim_procedure(
        self,
        ancestry: tuple[Component, ...] | None = None,
        current_order: dict[tuple[Spec, ...], BindingOptim] | None = None,
    ) -> dict[tuple[Spec, ...], BindingOptim]:
        # unfreeze eta during first step anywhere a round is an ancestor
        new_current_order = super().optim_procedure(ancestry, current_order)

        for key, binding_optim in new_current_order.items():
            if any(self in ancestors for ancestors in binding_optim.ancestry):
                new_current_order[key].steps[0].calls.append(
                    Call(self, "unfreeze", {"parameter": "eta"})
                )

        return new_current_order

    @abc.abstractmethod
    def log_aggregate(self, seqs: Tensor) -> Tensor:
        """Predict log aggregate (log Z_{i,r})"""

    @abc.abstractmethod
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        """Predict log enrichment ratio (log[f_{i,r} / f_{i,r-1}])"""

    @Transform.cache
    def log_cumulative_enrichment(self, seqs: Tensor) -> Tensor:
        """Predict log cumulative enrichment (Π_r f_{i,r})"""
        if (
            isinstance(self.reference_round, IRound)
            or self.reference_round is None
        ):
            return self.log_enrichment(seqs)
        return self.reference_round.log_cumulative_enrichment(
            seqs
        ) + self.log_enrichment(seqs)

    @override
    @Transform.cache
    def forward(self, seqs: Tensor) -> Tensor:
        """Predict relative log count (log[n_{i,r} Π_r f_{i,r}])"""
        return self.log_eta + self.log_cumulative_enrichment(seqs)


class _ARRound(_ARound):
    """R Round"""

    _unfreezable = Literal[_ARound._unfreezable, "rho"]

    def __init__(
        self,
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        library_concentration: float = -1,
        rho: float = 1,
        gamma: float = 0,
        name: str = "",
    ) -> None:
        _ARound.__init__(
            self,
            reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            name=name,
        )
        self._rho = torch.nn.Parameter(torch.tensor(rho, dtype=__precision__))
        self._gamma = torch.nn.Parameter(
            torch.tensor(gamma, dtype=__precision__)
        )

    @override
    def unfreeze(self, parameter: _unfreezable = "all") -> None:
        """Unfreeze the desired parameter"""
        if parameter in ("rho", "all"):
            self._rho.requires_grad_()
            self._gamma.requires_grad_()
        if parameter != "rho":
            super().unfreeze(parameter)

    @override
    def optim_procedure(
        self,
        ancestry: tuple[Component, ...] | None = None,
        current_order: dict[tuple[Spec, ...], BindingOptim] | None = None,
    ) -> dict[tuple[Spec, ...], BindingOptim]:
        # optimize rho after every step calling 'alpha_heuristic'
        current_order = super().optim_procedure(ancestry, current_order)

        for binding_optim in current_order.values():
            for step_idx, step in enumerate(binding_optim.steps):
                for call in step.calls:
                    if call.fun == "alpha_heuristic":
                        binding_optim.steps.insert(
                            step_idx + 1,
                            Step(
                                [Call(self, "unfreeze", {"parameter": "rho"})]
                            ),
                        )

        return current_order


class _AWRound(_ARound):
    """W Round"""

    _unfreezable = Literal[_ARound._unfreezable, "tau"]

    def __init__(
        self,
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        library_concentration: float = -1,
        tau: float = -2,
        eps: float = 2,
        name: str = "",
    ) -> None:
        _ARound.__init__(
            self,
            reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            name=name,
        )
        self._tau = torch.nn.Parameter(torch.tensor(tau, dtype=__precision__))
        self._eps = torch.nn.Parameter(torch.tensor(eps, dtype=__precision__))

    @override
    def unfreeze(self, parameter: _unfreezable = "all") -> None:
        """Unfreeze the desired parameter"""
        if parameter in ("tau", "all"):
            self._tau.requires_grad_()
            self._eps.requires_grad_()
        if parameter != "tau":
            super().unfreeze(parameter)

    @override
    def optim_procedure(
        self,
        ancestry: tuple[Component, ...] | None = None,
        current_order: dict[tuple[Spec, ...], BindingOptim] | None = None,
    ) -> dict[tuple[Spec, ...], BindingOptim]:
        # optimize tau after every step calling 'alpha_heuristic'
        current_order = super().optim_procedure(ancestry, current_order)

        for binding_optim in current_order.values():
            for step_idx, step in enumerate(binding_optim.steps):
                for call in step.calls:
                    if call.fun == "alpha_heuristic":
                        binding_optim.steps.insert(
                            step_idx + 1,
                            Step(
                                [Call(self, "unfreeze", {"parameter": "tau"})]
                            ),
                        )
            binding_optim.merge_binding_optim()

        return current_order

    def _get_log_tau(self, log_aggregate: Tensor) -> Tensor:
        tau = cast(Tensor, F.softplus(self._tau))
        log_eps = logsigmoid(self._eps)
        return torch.logaddexp(
            log1mexp(-log_eps) - (tau / torch.exp(log_aggregate)), log_eps
        )


class IRound(_ARound):
    """I Round"""

    def __init__(self, name: str = "") -> None:
        super().__init__(reference_round=None, train_eta=False, name=name)

    @override
    def components(self) -> Iterator[Aggregate]:
        return iter(())

    @override
    def log_aggregate(self, seqs: Tensor) -> Tensor:
        return self.log_enrichment(seqs)

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return torch.zeros(
            len(seqs), dtype=__precision__, device=self.log_eta.device
        )


class _Round(_ARound):
    """Default Round"""

    def __init__(
        self,
        aggregate: Aggregate,
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        library_concentration: float = -1,
        name: str = "",
    ) -> None:
        super().__init__(
            reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            name=name,
        )
        self.aggregate = aggregate
        self.reference_round: _ARound

    @classmethod
    def from_binding(
        cls,
        binding: Iterable[Binding],
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        train_concentration: bool = False,
        target_concentration: float = 1,
        library_concentration: float = -1,
        name: str = "",
    ) -> Self:
        return cls(
            Aggregate.from_binding(
                binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
            ),
            reference_round=reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            name=name,
        )

    @override
    def components(self) -> Iterator[Aggregate]:
        yield self.aggregate

    @override
    def log_aggregate(self, seqs: Tensor) -> Tensor:
        return self.aggregate(seqs)


class _RRound(_ARRound, _Round):
    """R Round"""

    def __init__(
        self,
        aggregate: Aggregate,
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        library_concentration: float = -1,
        name: str = "",
        rho: float = 1,
        gamma: float = 0,
    ) -> None:
        super().__init__(
            reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            rho=rho,
            gamma=gamma,
            name=name,
        )
        self.aggregate = aggregate
        self.reference_round: _ARound

    @override
    @classmethod
    def from_binding(
        cls,
        binding: Iterable[Binding],
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        train_concentration: bool = False,
        target_concentration: float = 1,
        library_concentration: float = -1,
        name: str = "",
        rho: float = 1,
        gamma: float = 0,
    ) -> Self:
        return cls(
            Aggregate.from_binding(
                binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
            ),
            reference_round=reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            rho=rho,
            gamma=gamma,
            name=name,
        )


class _WRound(_AWRound, _Round):
    """W Round"""

    def __init__(
        self,
        aggregate: Aggregate,
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        library_concentration: float = -1,
        name: str = "",
        tau: float = -2,
        eps: float = 2,
    ) -> None:
        super().__init__(
            reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            tau=tau,
            eps=eps,
            name=name,
        )
        self.aggregate = aggregate
        self.reference_round: _ARound

    @override
    @classmethod
    def from_binding(
        cls,
        binding: Iterable[Binding],
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        train_concentration: bool = False,
        target_concentration: float = 1,
        library_concentration: float = -1,
        name: str = "",
        tau: float = -2,
        eps: float = 2,
    ) -> Self:
        return cls(
            Aggregate.from_binding(
                binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
            ),
            reference_round=reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            tau=tau,
            eps=eps,
            name=name,
        )


class BRound(_Round):
    """B Round"""

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return logsigmoid(self.log_aggregate(seqs))


class BSRound(_Round):
    """BS Round"""

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return logsigmoid(logsigmoid(self.log_aggregate(seqs)))


class BRRound(_RRound):
    """BR Round"""

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        log_agg = self.log_aggregate(seqs)
        return self.rho * log_agg - self.gamma * cast(
            Tensor, F.softplus(log_agg)
        )


class BWRound(_WRound, BRound):
    """BW Round"""

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return super().log_enrichment(seqs) + self._get_log_tau(
            self.log_aggregate(seqs)
        )


class BURound(_Round):
    """BU Round"""

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return self.log_aggregate(seqs)


class BUWRound(_WRound, BURound):
    """BUW Round"""

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return super().log_enrichment(seqs) + self._get_log_tau(
            self.log_aggregate(seqs)
        )


class FRound(_Round):
    """F Round"""

    @classmethod
    def from_round(cls, base_round: _Round, name: str = "") -> Self:
        """Create FRound with same aggregates, reference_round,
        and train_eta as base_round
        """
        return cls(
            base_round.aggregate,
            base_round.reference_round,
            train_eta=base_round.train_eta,
            library_concentration=base_round.library_concentration.item(),
            name=name,
        )

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return cast(Tensor, -F.softplus(self.log_aggregate(seqs)))


class ERound(_Round):
    """E Round"""

    _unfreezable = Literal[_Round._unfreezable, "delta"]

    def __init__(
        self,
        aggregate: Aggregate,
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        library_concentration: float = -1,
        delta: float = -15,
        train_delta: bool = True,
        name: str = "",
    ) -> None:
        self.reference_round: _ARound
        super().__init__(
            aggregate,
            reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            name=name,
        )
        self._delta = torch.nn.Parameter(
            torch.tensor(delta, dtype=__precision__)
        )
        self.train_delta = train_delta

    @classmethod
    def from_round(
        cls,
        base_round: Self,
        reference_round: _ARound,
        target_concentration: float = 1,
        name: str = "",
    ) -> Self:
        """Create ERound with same contributions and delta"""
        return cls(
            Aggregate(
                base_round.aggregate.contributions,
                train_concentration=base_round.aggregate.train_concentration,
                target_concentration=target_concentration,
            ),
            reference_round,
            train_eta=base_round.train_eta,
            delta=base_round._delta.item(),  # pylint: disable=protected-access
            train_delta=base_round.train_delta,
            name=name,
        )

    @override
    @classmethod
    def from_binding(
        cls,
        binding: Iterable[Binding],
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        train_concentration: bool = False,
        target_concentration: float = 1,
        library_concentration: float = -1,
        name: str = "",
        delta: float = -15,
        train_delta: bool = True,
    ) -> Self:
        return cls(
            Aggregate.from_binding(
                binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
            ),
            reference_round=reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            delta=delta,
            train_delta=train_delta,
            name=name,
        )

    @override
    def unfreeze(self, parameter: _unfreezable = "all") -> None:
        """Unfreeze the desired parameter"""
        if parameter in ("delta", "all") and self.train_delta:
            self._delta.requires_grad_()
        if parameter != "delta":
            super().unfreeze(parameter)

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        aggregate = torch.exp(self.log_aggregate(seqs))
        if not torch.isneginf(self._delta):
            return torch.logaddexp(
                logsigmoid(self._delta) - aggregate,
                logsigmoid(-self._delta) + log1mexp(aggregate),
            )
        return log1mexp(aggregate)


class _CRound(_ARound):
    """C Round"""

    def __init__(
        self,
        s_aggregate: Aggregate,
        c_aggregate: Aggregate,
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        library_concentration: float = -1,
        name: str = "",
    ) -> None:
        self.reference_round: _ARound
        super().__init__(
            reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            name=name,
        )
        self.s_aggregate = s_aggregate
        self.c_aggregate = c_aggregate

    @classmethod
    def from_binding(
        cls,
        s_binding: Iterable[Binding],
        c_binding: Iterable[Binding],
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        train_concentration: bool = False,
        target_concentration: float = 1,
        library_concentration: float = -1,
        name: str = "",
    ) -> Self:
        return cls(
            Aggregate.from_binding(
                s_binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
            ),
            Aggregate.from_binding(
                c_binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
            ),
            reference_round=reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            name=name,
        )

    @override
    def components(self) -> Iterator[Aggregate]:
        return iter((self.s_aggregate, self.c_aggregate))

    @override
    def log_aggregate(self, seqs: Tensor) -> Tensor:
        return self.s_aggregate(seqs) - cast(
            Tensor, F.softplus(self.c_aggregate(seqs))
        )


class _WCRound(_AWRound, _CRound):
    """WC Round"""

    def __init__(
        self,
        s_aggregate: Aggregate,
        c_aggregate: Aggregate,
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        library_concentration: float = -1,
        tau: float = -2,
        eps: float = 2,
        name: str = "",
    ) -> None:
        super().__init__(
            reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            tau=tau,
            eps=eps,
            name=name,
        )
        self.s_aggregate = s_aggregate
        self.c_aggregate = c_aggregate

    @override
    @classmethod
    def from_binding(
        cls,
        s_binding: Iterable[Binding],
        c_binding: Iterable[Binding],
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        train_concentration: bool = False,
        target_concentration: float = 1,
        library_concentration: float = -1,
        name: str = "",
        tau: float = -2,
        eps: float = 2,
    ) -> Self:
        return cls(
            Aggregate.from_binding(
                s_binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
            ),
            Aggregate.from_binding(
                c_binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
            ),
            reference_round=reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            tau=tau,
            eps=eps,
            name=name,
        )


class _RCRound(_ARRound, _CRound):
    """RC Round"""

    def __init__(
        self,
        s_aggregate: Aggregate,
        c_aggregate: Aggregate,
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        library_concentration: float = -1,
        rho: float = 1,
        gamma: float = 0,
        name: str = "",
    ) -> None:
        super().__init__(
            reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            rho=rho,
            gamma=gamma,
            name=name,
        )
        self.s_aggregate = s_aggregate
        self.c_aggregate = c_aggregate

    @override
    @classmethod
    def from_binding(
        cls,
        s_binding: Iterable[Binding],
        c_binding: Iterable[Binding],
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        train_concentration: bool = False,
        target_concentration: float = 1,
        library_concentration: float = -1,
        name: str = "",
        rho: float = 1,
        gamma: float = 0,
    ) -> Self:
        return cls(
            Aggregate.from_binding(
                s_binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
            ),
            Aggregate.from_binding(
                c_binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
            ),
            reference_round=reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            rho=rho,
            gamma=gamma,
            name=name,
        )

    @override
    def log_aggregate(self, seqs: Tensor) -> Tensor:
        return self.rho * self.s_aggregate(seqs) - self.gamma * cast(
            Tensor, F.softplus(self.c_aggregate(seqs))
        )


class BCRound(_CRound):
    """BC Round"""

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return logsigmoid(self.log_aggregate(seqs))


class BRCRound(_CRound):
    """BRC Round"""

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return logsigmoid(self.log_aggregate(seqs))


class BWCRound(_WCRound, BCRound):
    """BWC Round"""

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return super().log_enrichment(seqs) + self._get_log_tau(
            self.s_aggregate(seqs)
        )


class _SRound(_ARound):
    """C Round"""

    def __init__(
        self,
        i_aggregate: Aggregate,
        f_aggregate: Aggregate,
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        library_concentration: float = -1,
        name: str = "",
    ) -> None:
        self.reference_round: _ARound
        super().__init__(
            reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            name=name,
        )
        self.i_aggregate = i_aggregate
        self.f_aggregate = f_aggregate

    @classmethod
    def from_binding(
        cls,
        i_binding: Iterable[Binding],
        f_binding: Iterable[Binding],
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        train_concentration: bool = False,
        target_concentration: float = 1,
        library_concentration: float = -1,
        name: str = "",
    ) -> Self:
        return cls(
            Aggregate.from_binding(
                i_binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
            ),
            Aggregate.from_binding(
                f_binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
            ),
            reference_round=reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            name=name,
        )

    @override
    def components(self) -> Iterator[Aggregate]:
        return iter((self.i_aggregate, self.f_aggregate))

    @override
    def log_aggregate(self, seqs: Tensor) -> Tensor:
        return torch.logaddexp(
            self.i_aggregate(seqs), (self.f_aggregate(seqs))
        )


class SRound(_SRound):
    """S Round"""

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return logsigmoid(self.log_aggregate(seqs))


class SURound(_SRound):
    """SU Round"""

    @override
    def log_enrichment(self, seqs: Tensor) -> Tensor:
        return self.log_aggregate(seqs)


class _KRound(_ARound):
    """K Round"""

    def __init__(
        self,
        f_aggregate: Aggregate,
        t_aggregate: Aggregate,
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        library_concentration: float = -1,
        name: str = "",
    ) -> None:
        if f_aggregate.train_concentration:
            raise ValueError(
                "Only the t_aggregate can have train_concentration"
            )
        super().__init__(
            reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            name=name,
        )
        self.f_aggregate = f_aggregate
        self.t_aggregate = t_aggregate

    @classmethod
    def from_binding(
        cls,
        f_binding: Iterable[Binding],
        t_binding: Iterable[Binding],
        reference_round: _ARound,
        train_eta: bool = True,
        log_eta: float = 0,
        train_concentration: bool = False,
        target_concentration: float = 1,
        library_concentration: float = -1,
        name: str = "",
    ) -> Self:
        return cls(
            Aggregate.from_binding(
                f_binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
            ),
            Aggregate.from_binding(
                t_binding,
                train_concentration=train_concentration,
                target_concentration=target_concentration,
            ),
            reference_round=reference_round,
            train_eta=train_eta,
            log_eta=log_eta,
            library_concentration=library_concentration,
            name=name,
        )

    @override
    def log_aggregate(self, seqs: Tensor) -> Tensor:
        return self.t_aggregate(seqs) - self.f_aggregate(seqs)
