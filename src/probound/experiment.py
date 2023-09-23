"""Binding and selection models"""
from collections.abc import Iterable, Iterator

import torch
from numpy.typing import ArrayLike
from torch import Tensor
from typing_extensions import override

from . import __precision__
from .base import Transform
from .containers import Buffer, TModuleList
from .rounds import BRound, FRound, _ARound


class Experiment(Transform):
    """Sequence of sequenced rounds corresponding to count table columns"""

    def __init__(
        self,
        rounds: Iterable[_ARound],
        counts_per_round: ArrayLike | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name)

        # store instance attributes
        self.observed_rounds = list(rounds)
        if all(rnd.train_eta for rnd in self.observed_rounds):
            raise ValueError("At least one round must have train_eta=False")
        if len(self.observed_rounds) != len(set(self.observed_rounds)):
            raise ValueError("Cannot repeat the same round twice")

        # get all rounds (including unobserved)
        # use dict to preserve insertion order and uniqueness
        round_ancestry = {rnd: None for rnd in self.observed_rounds}
        for rnd in self.observed_rounds:
            cur_rnd = rnd
            while cur_rnd.reference_round is not None:
                round_ancestry[cur_rnd] = None
                cur_rnd = cur_rnd.reference_round
        self.rounds = TModuleList(round_ancestry.keys())

        # check eta training
        for rnd in self.rounds:
            if rnd not in self.observed_rounds and rnd.train_eta:
                raise ValueError(f"{rnd} has train_eta=True but is unobserved")

        # register counts_per_round
        if counts_per_round is None:
            counts_per_round = [-1] * len(self.rounds)
        self._counts_per_round: Tensor = Buffer(
            torch.tensor(counts_per_round, dtype=__precision__)
            if not isinstance(counts_per_round, Tensor)
            else counts_per_round.to(__precision__)
        )

    @property
    def counts_per_round(self) -> Tensor:
        if torch.any(self._counts_per_round < 0):
            raise ValueError(f"{self} not initialized with 'counts_per_round'")
        return self._counts_per_round

    @override
    def components(self) -> Iterator[_ARound]:
        return iter(self.rounds)

    @override
    @Transform.cache
    def forward(self, seqs: Tensor) -> Tensor:
        """Predict log probe frequencies
        (log [n_{i,r} f_{i,r} / sum_r' n_{i,r'} f_{i,r'}])
        """
        out = torch.stack([rnd(seqs) for rnd in self.observed_rounds], dim=1)
        return out - out.logsumexp(dim=1, keepdim=True)

    def log_prediction(self, seqs: Tensor, target: Tensor) -> Tensor:
        """Predict log count table (k_{i,r})"""
        log_frequencies = self(seqs)
        if log_frequencies.shape != target.shape:
            raise ValueError(
                f"Predicted table shape {log_frequencies.shape}"
                f" incompatible with observed table shape {target.shape}"
            )
        return log_frequencies + torch.log(
            torch.sum(target, dim=1, keepdim=True)
        )

    def free_protein(
        self,
        i_index: int,
        b_index: int,
        f_index: int,
        target_concentration: float | None = None,
        library_concentration: float | None = None,
    ) -> float:
        """Calculate free protein concentration"""
        i_round = self.rounds[i_index]
        b_round = self.rounds[b_index]
        f_round = self.rounds[f_index]
        if not isinstance(b_round, BRound):
            raise ValueError(f"Round at index {b_round} is not {BRound}")
        if not isinstance(f_round, FRound):
            raise ValueError(f"Round at index {f_round} is not {FRound}")
        if b_round.reference_round != i_round:
            raise ValueError(f"reference_round of {b_round} is not {i_round}")
        if f_round.reference_round != i_round:
            raise ValueError(f"reference_round of {f_round} is not {i_round}")
        if b_round.aggregate != f_round.aggregate:
            raise ValueError(
                f"Rounds {b_round}, {f_round} do not share an aggregate"
            )

        if target_concentration is None:
            target_concentration = torch.exp(
                b_round.aggregate.log_target_concentration
            ).item()
        if library_concentration is None:
            library_concentration = b_round.library_concentration.item()

        with torch.inference_mode():
            prob_bound = (
                self.counts_per_round[b_index] / self.counts_per_round[i_index]
            ) * torch.exp(
                self.rounds[i_index].log_eta - self.rounds[b_index].log_eta
            )
        return target_concentration - prob_bound.item() * library_concentration
