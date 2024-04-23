"""Experiment class modeling a CountTable.

Members are explicitly re-exported in pyprobound.
"""

import graphlib
from collections.abc import Iterable, Iterator

import torch
from torch import Tensor
from typing_extensions import override

from . import __precision__
from .base import Transform
from .containers import TModuleList
from .rounds import BaseRound, BoundRound, UnboundRound


class Experiment(Transform):
    """Models sequenced rounds corresponding to count table columns.

    Attributes:
        observed_rounds (list[BaseRound]): The sequenced rounds modeled by the
            experiment.
        rounds (TModuleList[BaseRound]): A ModuleList of all rounds, including
            those not sequenced.
    """

    def __init__(
        self,
        rounds: Iterable[BaseRound],
        counts_per_round: Tensor | list[float] | None = None,
        name: str = "",
    ) -> None:
        r"""Initializes the experiment from an iterable of sequenced rounds.

        Args:
            rounds: The sequenced rounds modeled by the experiment.
            counts_per_round: A tensor with the number of probes in each round
                of the count table used for training, with shape
                :math:`(\text{rounds},)`. Should be provided for Kd-seq.
            name: A string used to describe the experiment.
        """

        super().__init__(name=name)

        # Store instance attributes
        self.observed_rounds = list(rounds)
        if all(rnd.train_depth for rnd in self.observed_rounds):
            raise ValueError("At least one round must have train_depth=False")
        if len(self.observed_rounds) != len(set(self.observed_rounds)):
            raise ValueError("Cannot repeat the same round twice")

        # Get all rounds (including unobserved)
        sorter: graphlib.TopologicalSorter[BaseRound] = (
            graphlib.TopologicalSorter()
        )
        for rnd in self.observed_rounds:
            while rnd.reference_round is not None:
                sorter.add(rnd, rnd.reference_round)
                rnd = rnd.reference_round
        self.rounds: TModuleList[BaseRound] = TModuleList(
            sorter.static_order()
        )

        # Check depth training
        for rnd in self.rounds:
            if rnd not in self.observed_rounds and rnd.train_depth:
                raise ValueError(
                    f"{rnd} has train_depth=True but is unobserved"
                )

        # Register counts_per_round
        if counts_per_round is None:
            counts_per_round = [-1 for _ in enumerate(self.rounds)]
        if isinstance(counts_per_round, Tensor):
            counts_per_round = counts_per_round.tolist()
        self._counts_per_round = counts_per_round

    @property
    def counts_per_round(self) -> list[float]:
        r"""A list of the number of probes in each round of the count table
        used for training, with shape :math:`(\text{rounds},)`.
        """
        if any(i < 0 for i in self._counts_per_round):
            raise ValueError(f"{self} not initialized with 'counts_per_round'")
        return self._counts_per_round

    @override
    def components(self) -> Iterator[BaseRound]:
        return iter(self.rounds)

    @override
    @Transform.cache
    def forward(self, seqs: Tensor) -> Tensor:
        r"""Predicts the log probe frequencies.

        .. math::
            \log \frac{\eta_{r} f_{i,r}}{
                \sum_{r^\prime} \eta_{r^\prime} f_{i, r^\prime}
            }

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            The log frequency tensor of shape
            :math:`(\text{minibatch},\text{rounds})`.
        """
        out = torch.stack([rnd(seqs) for rnd in self.observed_rounds], dim=1)
        return out - out.logsumexp(dim=1, keepdim=True)

    def free_protein(
        self,
        i_index: int,
        b_index: int,
        u_index: int,
        target_concentration: float | None = None,
        library_concentration: float | None = None,
    ) -> float:
        r"""Estimates the free protein concentration.

        If the input (I), bound (B), and unbound (U) probes of a selection are
        modeled jointly, PyProBound can estimate free protein concentration.

        .. math::
            [\text{P}]_F &= [\text{P}]_T - [\text{library}] p(B) \\
            p(B) &= \frac{k_B}{k_I} \frac{\eta_I}{\eta_B}

        Args:
            i_index: The index of the input round.
            b_index: The index of the BoundRound.
            u_index: The index of the UnboundRound.
            target_concentration: The total protein concentration
                :math:`[\text{P}]_T`, taken from Aggregate if not provided.
            library_concentration: The total library concentration
                :math:`[\text{library}]`, taken from Round if not provided.

        Returns:
            The free protein concentration represented as a float.
        """
        i_round = self.rounds[i_index]
        b_round = self.rounds[b_index]
        u_round = self.rounds[u_index]
        if not isinstance(b_round, BoundRound):
            raise ValueError(f"Round at index {b_round} is not {BoundRound}")
        if not isinstance(u_round, UnboundRound):
            raise ValueError(f"Round at index {u_round} is not {UnboundRound}")
        if b_round.reference_round != i_round:
            raise ValueError(f"reference_round of {b_round} is not {i_round}")
        if u_round.reference_round != i_round:
            raise ValueError(f"reference_round of {u_round} is not {i_round}")
        if b_round.aggregate != u_round.aggregate:
            raise ValueError(
                f"Rounds {b_round}, {u_round} do not share an aggregate"
            )

        if target_concentration is None:
            target_concentration = torch.exp(
                b_round.aggregate.log_target_concentration
            ).item()
        if library_concentration is None:
            library_concentration = b_round.library_concentration

        with torch.inference_mode():
            prob_bound = (
                self.counts_per_round[b_index] / self.counts_per_round[i_index]
            ) * torch.exp(
                self.rounds[i_index].log_depth - self.rounds[b_index].log_depth
            )
        return target_concentration - prob_bound.item() * library_concentration
