"""Module for loading and scoring count tables and sequences"""
import abc
import dataclasses
import functools
import gzip
import itertools
import os
from collections.abc import Callable, Iterable, Iterator, Sized
from typing import Any, Generic, Protocol, TypeVar, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler, SubsetRandomSampler
from typing_extensions import override

from . import __precision__
from .alphabet import Alphabet
from .base import Component
from .utils import get_split_size

T = TypeVar("T")
PSAM = Any


class Batch(Protocol):
    seqs: Tensor
    target: Tensor


@dataclasses.dataclass
class BatchTuple(Batch):
    seqs: Tensor
    target: Tensor


def score(
    module: Component, batch: Batch, fun: str = "forward", **kwargs: Any
) -> tuple[Tensor, Tensor]:
    """(obs, pred) on CPU from batch using specified module function"""
    method: Callable[[Tensor], Tensor] = getattr(module, fun)
    for p in module.parameters():
        device = p.device
        break
    split_size = get_split_size(
        module.max_embedding_size(), len(batch.seqs), device
    )

    predictions: list[Tensor] = []
    observations: list[Tensor] = []
    with torch.inference_mode():
        module.eval()
        for seqs, target in zip(
            torch.split(batch.seqs, split_size),
            torch.split(batch.target, split_size),
        ):
            predictions.append(method(seqs.to(device), **kwargs).cpu())
            observations.append(target.cpu())

    return torch.cat(observations), torch.cat(predictions)


def get_dataframe(
    paths: list[str],
    total_count: int | None = None,
    random_state: int | None = None,
) -> DataFrame:
    """Loads a tab-delimited count table into a dataframe"""

    if not isinstance(paths, list):
        raise TypeError(
            "paths argument to get_dataframe should be a list of str,"
            " each a path to a different tsv"
        )

    dataframes: list[DataFrame] = []
    idx = 0
    for idx, path in enumerate(paths):
        open_fn = gzip.open if os.path.splitext(path)[-1] == ".gz" else open
        with open_fn(path, "rt", encoding="utf-8") as file_handle:  # type: ignore[operator]
            first_line = file_handle.readline()
            skiprows = 0 if "\t" in first_line else 1

        df = pd.read_csv(
            path, header=None, index_col=0, sep="\t", skiprows=skiprows
        )
        if total_count is not None:
            df = sample_counts(df, total_count, random_state)

        columns = range(idx, idx + len(df.columns))
        idx += len(df.columns)
        df = df.set_axis(columns, axis=1)
        dataframes.append(df)

    return functools.reduce(
        lambda df1, df2: df1.join(df2, how="outer").fillna(0).astype(int),
        dataframes,
    )


def sample_dataframe(
    dataframe: DataFrame,
    frac: float = 0.1,
    random_state: int | None = None,
    n_bin: int = 128,
) -> tuple[DataFrame, DataFrame]:
    """Sample from a dataframe evenly by counts per row"""
    dataframe["Enrichment"] = (dataframe.iloc[:, -1] + 1) / (
        dataframe.iloc[:, 0] + 1
    )
    dataframe = dataframe.sort_values("Enrichment")

    samples: list[DataFrame] = []
    step = len(dataframe) // n_bin
    for i in range(n_bin):
        if i < (n_bin - 1):
            binned = dataframe.iloc[step * i : step * (i + 1)]
        else:
            binned = dataframe.iloc[step * i :]
        samples.append(binned.sample(frac=frac, random_state=random_state))

    sample = pd.concat(samples).drop(columns="Enrichment").sort_index()
    return (
        sample,
        dataframe.drop(columns="Enrichment").drop(sample.index).sort_index(),
    )


def sample_counts(
    dataframe: DataFrame,
    n_counts: int = 1_000_000,
    random_state: int | None = None,
) -> DataFrame:
    """Randomly split a count table into n different tables"""
    total = dataframe.iloc[:, 0].sum()
    generator = np.random.default_rng(random_state)
    dataframe.iloc[:, 0] = generator.binomial(
        dataframe.iloc[:, 0], min(1, n_counts / total)
    )
    return dataframe[dataframe.iloc[:, 0] != 0]


class Table(Dataset[T], Generic[T], Sized, abc.ABC):
    def __init__(
        self, left_flank_length: int = 0, right_flank_length: int = 0
    ) -> None:
        self.left_flank_length = 0
        self.right_flank_length = 0
        self.set_flank_length(left=left_flank_length, right=right_flank_length)

    @property
    @abc.abstractmethod
    def input_shape(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def min_read_length(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def max_read_length(self) -> int:
        ...

    @abc.abstractmethod
    def get_setup_string(self) -> str:
        ...

    @abc.abstractmethod
    def set_flank_length(self, left: int = 0, right: int = 0) -> None:
        ...


class CountTable(Table[Batch]):
    """Loads pandas count table into tensor"""

    def __init__(
        self,
        dataframe: DataFrame,
        alphabet: Alphabet,
        transliterate: dict[str, str] | None = None,
        left_flank: str = "",
        right_flank: str = "",
        left_flank_length: int = 0,  # current left flank to be used
        right_flank_length: int = 0,  # current right flank to be used
        max_left_flank_length: int
        | None = None,  # max length of left flank (default=len(left_flank))
        max_right_flank_length: int
        | None = None,  # max length of right flank (default=len(right_flank))
        zero_pad: bool = False,  # zero-pad all sequences to be the same length
        min_variable_length: int
        | None = None,  # min variable length (default=min(df.index.str.len()))
        max_variable_length: int
        | None = None,  # max variable length (default=max(df.index.str.len()))
    ) -> None:
        # transliterate if necessary
        if transliterate is not None:
            for pattern, replace in transliterate.items():
                dataframe.index = dataframe.index.str.replace(pattern, replace)
                left_flank = left_flank.replace(pattern, replace)
                right_flank = right_flank.replace(pattern, replace)

        # instance attributes
        self.alphabet = alphabet
        self.left_flank = left_flank
        self.right_flank = right_flank
        if max_left_flank_length is None:
            max_left_flank_length = len(left_flank)
        if max_right_flank_length is None:
            max_right_flank_length = len(right_flank)
        self.max_left_flank_length = max_left_flank_length
        self.max_right_flank_length = max_right_flank_length
        self.zero_padded = zero_pad

        # store variable lengths
        self.variable_lengths = (
            torch.tensor(
                dataframe.index.str.len() - dataframe.index.str.count(r"\+")
            ).unsqueeze(1)
            // alphabet.monomer_length
        )
        curr_min_variable_length = (
            self.variable_lengths.min().item() // alphabet.monomer_length
        )
        curr_max_variable_length = (
            self.variable_lengths.max().item() // alphabet.monomer_length
        )
        if min_variable_length is None:
            min_variable_length = cast(int, curr_min_variable_length)
        if max_variable_length is None:
            max_variable_length = cast(int, curr_max_variable_length)
        if curr_min_variable_length < min_variable_length:
            raise ValueError(
                "min_variable_length is smaller than"
                " the shortest sequence in dataframe"
            )
        if curr_max_variable_length > max_variable_length:
            raise ValueError(
                "max_variable_length is smaller than"
                " the longest sequence in dataframe"
            )
        # needed for omega + theta if train/val/test splitting
        self.min_variable_length = min_variable_length
        self.max_variable_length = max_variable_length

        # get dataframe data
        self.padding_value = alphabet.get_index["*" if zero_pad else " "]
        self.target = torch.tensor(dataframe.values, dtype=__precision__)
        self.seqs = torch.nn.utils.rnn.pad_sequence(
            [alphabet.translate(seq) for seq in dataframe.index],
            batch_first=True,
            padding_value=self.padding_value,
        )
        self.seqs = F.pad(
            self.seqs,
            (0, self.max_variable_length - self.seqs.shape[1]),
            value=self.padding_value,
        ).contiguous()

        # get number of probes per round
        self.counts_per_round = torch.sum(self.target, dim=0)

        # set flank length
        super().__init__(
            left_flank_length=left_flank_length,
            right_flank_length=right_flank_length,
        )

    @override
    @property
    def input_shape(self) -> int:
        return self.seqs.shape[-1]

    @override
    @property
    def min_read_length(self) -> int:
        if self.zero_padded:
            return self.max_read_length
        return (
            self.min_variable_length
            + self.left_flank_length
            + self.right_flank_length
        )

    @override
    @property
    def max_read_length(self) -> int:
        return (
            self.max_variable_length
            + self.left_flank_length
            + self.right_flank_length
        )

    @override
    def set_flank_length(
        self,
        left: int = 0,  # new length of left flank
        right: int = 0,  # new length of right flank
    ) -> None:
        """Set the length of flanks included in self.seqs"""

        # check input
        if left > self.max_left_flank_length:
            raise ValueError(
                f"left flank length of {left} exceeds"
                f" max_left_flank_length of {self.max_left_flank_length}"
            )
        if right > self.max_right_flank_length:
            raise ValueError(
                f"right flank length of {right} exceeds"
                f" max_right_flank_length of {self.max_right_flank_length}"
            )
        if left < 0 or right < 0:
            raise ValueError("Flank lengths must be nonnegative")
        if self.left_flank_length == left and self.right_flank_length == right:
            return

        # trim new flanks to desired lengths
        old_left_flank = self.left_flank[
            len(self.left_flank) - self.left_flank_length :
        ]
        left_flank = self.left_flank.rjust(left, "*")[
            len(self.left_flank) - left :
        ]
        old_left_flank_tr = self.alphabet.translate(old_left_flank)
        left_flank_tr = self.alphabet.translate(left_flank)
        old_right_flank = self.right_flank[: self.right_flank_length]
        right_flank = self.right_flank.ljust(right, "*")[:right]
        right_flank_tr = self.alphabet.translate(right_flank)
        old_right_flank_tr = self.alphabet.translate(old_right_flank)

        ### update left flank
        if len(left_flank_tr) < len(old_left_flank_tr):
            # trimming left flank
            self.seqs = self.seqs[
                :, len(old_left_flank_tr) - len(left_flank_tr) :
            ]
        elif left > self.left_flank_length:
            # extending left flank
            left_update = left_flank_tr[
                : len(left_flank_tr) - len(old_left_flank_tr)
            ]
            self.seqs = torch.hstack(
                (left_update.expand(len(self), -1), self.seqs)
            )
        self.left_flank_length = left

        ### update right flank
        if len(right_flank_tr) < len(old_right_flank_tr):
            # trimming right flank
            diff = len(old_right_flank_tr) - len(right_flank_tr)
            indices = (
                self.variable_lengths.expand(len(self), diff)
                + torch.arange(diff)
                + len(right_flank_tr)
                + len(left_flank_tr)
            )
            values = torch.full(
                (diff,), self.alphabet.neginf_pad, dtype=self.seqs.dtype
            )
            self.seqs = self.seqs.scatter_(
                1, indices, values.expand(len(self), -1)
            )
            self.seqs = self.seqs[:, :-diff]
        else:
            # extending right flank
            diff = len(right_flank_tr) - len(old_right_flank_tr)
            self.seqs = F.pad(self.seqs, (0, diff), value=self.padding_value)
            indices = (
                self.variable_lengths.expand(len(self), diff)
                + torch.arange(diff)
                + len(old_right_flank_tr)
                + len(left_flank_tr)
            )
            right_update = right_flank_tr[
                len(old_right_flank_tr) : len(right_flank_tr)
            ]
            self.seqs = self.seqs.scatter_(
                1, indices, right_update.expand(len(self), -1)
            )
        self.right_flank_length = right

        # make contiguous
        self.seqs = self.seqs.contiguous()

    def prob_bound(self) -> Tensor:
        """Calculates p(b), assuming columns arranged as Input-Bound-Free"""
        if self.target.shape[1] != 3:
            raise ValueError("Must have 3 count columns (Input-Bound-Free)")
        cov_mat = torch.cov(self.target.T, correction=0)
        cov_dict = {}
        for i, i_name in enumerate("IBU"):
            for j, j_name in enumerate("IBU"):
                cov_dict[i_name + j_name] = cov_mat[i, j]
        return (
            cov_dict["BU"] - cov_dict["IB"] + cov_dict["IU"] - cov_dict["UU"]
        ) / (2 * cov_dict["BU"] - cov_dict["BB"] - cov_dict["UU"])

    def __delitem__(self, idx: int | slice | Tensor) -> None:
        inv_idx: list[int] | Tensor
        if isinstance(idx, int):
            inv_idx = list(range(idx)) + list(range(idx + 1, len(self)))
        elif isinstance(idx, slice):
            inv_idx = [
                i
                for i in range(len(self))
                if i not in range(*idx.indices(len(self)))
            ]
        elif isinstance(idx, torch.BoolTensor):
            inv_idx = ~idx
        else:
            unique, counts = torch.unique(
                torch.cat((idx, torch.arange(0, len(self))), dim=0)
            )
            inv_idx = cast(Tensor, unique[counts == 1])
        self.seqs = self.seqs[inv_idx].contiguous()
        self.target = self.target[inv_idx].contiguous()
        self.variable_lengths = self.variable_lengths[inv_idx].contiguous()
        self.min_variable_length = cast(
            int, self.variable_lengths.min().item()
        )
        self.max_variable_length = cast(
            int, self.variable_lengths.max().item()
        )
        self.counts_per_round = torch.sum(self.target, dim=0)

    @override
    def __getitem__(self, idx: int) -> BatchTuple:
        return BatchTuple(self.seqs[idx], self.target[idx])

    @override
    def __len__(self) -> int:
        return len(self.seqs)

    @override
    def get_setup_string(self) -> str:
        return "\n".join(
            [
                f"\t\tMaximum Variable Length: {self.max_variable_length}",
                f"\t\tLeft Flank Length: {self.left_flank_length}",
                f"\t\tRight Flank Length: {self.right_flank_length}",
            ]
        )

    def leverage_dict(
        self, mincount: int = 1, eps: float = 1e-8
    ) -> tuple["PSAM", DataFrame]:
        # TODO: rewrite
        def get_nddg(mask: Tensor) -> float:
            target = self.target[mask] / self.target.sum(0, keepdim=True)
            return -torch.log(
                torch.max(torch.sum(target[:, 0]), torch.tensor(eps))
                / torch.max(torch.sum(target[:, -1]), torch.tensor(eps))
            ).item()

        psam = PSAM(
            kernel_size=self.max_read_length,
            alphabet=self.alphabet,
            interaction_distance=self.max_read_length - 2,
        )
        for param in psam.betas.values():
            torch.nn.init.zeros_(param)
        leverage_dict = {}

        for pos1 in range(self.max_read_length):
            for idx1, alpha1 in enumerate(self.alphabet.alphabet):
                # mono
                mask1 = self.seqs[:, pos1] == idx1
                if (
                    mincount
                    < torch.sum(self.target[mask1])
                    < torch.sum(self.counts_per_round) - mincount
                ):
                    mask1_nddg = get_nddg(mask1)
                    leverage_dict[f"{pos1}{alpha1}"] = {
                        "n": mask1.sum().item(),
                        "nddg": mask1_nddg,
                    }
                else:
                    mask1_nddg = 0.0

                psam.betas[
                    PSAM._get_key(  # pylint: disable=protected-access
                        (torch.tensor(pos1 + 1), torch.tensor(pos1 + 1)), idx1
                    )
                ] = torch.nn.Parameter(torch.tensor(mask1_nddg))

                for pos2 in range(pos1 + 1, self.max_read_length):
                    for idx2, alpha2 in enumerate(self.alphabet.alphabet):
                        # di
                        mask2 = self.seqs[:, pos2] == idx2
                        mask = mask1 & mask2
                        if (
                            mincount
                            < torch.sum(self.target[mask])
                            < torch.sum(self.counts_per_round) - mincount
                        ):
                            mask2_nddg = get_nddg(mask2)
                            mask_nddg = get_nddg(mask)
                            # nddg = mask_nddg - mask1_nddg - mask2_nddg
                            nddg = mask1_nddg + mask2_nddg - mask_nddg
                            leverage_dict[f"{pos1}{alpha1}-{pos2}{alpha2}"] = {
                                "n": mask.sum().item(),
                                "nddg": nddg,
                            }
                        else:
                            nddg = 0.0

                        psam.betas[
                            PSAM._get_key(  # pylint: disable=protected-access
                                (
                                    torch.tensor(pos1 + 1),
                                    torch.tensor(pos2 + 1),
                                ),
                                idx1 * len(self.alphabet.alphabet) + idx2,
                            )
                        ] = torch.nn.Parameter(torch.tensor(nddg))

        leverage_dataframe = pd.DataFrame.from_dict(
            leverage_dict, orient="index", columns=["n", "nddg"]
        )
        return psam, leverage_dataframe.reindex(
            leverage_dataframe.nddg.abs().sort_values(ascending=False).index
        )


class EvenSampler(Sampler[int]):
    """Evenly sample across the range of indices"""

    def __init__(self, data_source: Sized, n_bin: int = 128) -> None:
        self.data_source = data_source
        self.n_bin = n_bin

    @override
    def __iter__(self) -> Iterator[int]:
        samplers = [
            iter(SubsetRandomSampler(bin))
            for bin in np.array_split(np.arange(len(self)), self.n_bin)
        ]
        for sampler in itertools.cycle(samplers):
            try:
                yield next(sampler)
            except StopIteration:
                return

    def __len__(self) -> int:
        return len(self.data_source)


class MultitaskLoader(Generic[T]):
    """Combines multiple dataloaders for multitask learning"""

    def __init__(self, dataloaders: Iterable[DataLoader[T]]) -> None:
        self.dataloaders = dataloaders
        self._longest_loader = max(self.dataloaders, key=len)

    @classmethod
    def cycle(cls, dataloader: DataLoader[T]) -> Iterator[T]:
        """Loop infinitely through an iterable"""
        iterator = iter(dataloader)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(dataloader)

    def __iter__(self) -> Iterator[tuple[T, ...]]:
        """Each item is a tuple of mini-batches from each dataloader"""

        if all(len(loader) == 1 for loader in self.dataloaders):
            yield tuple(cast(T, loader.dataset) for loader in self.dataloaders)
            return

        loaders: list[Iterator[T]] = []
        for loader in self.dataloaders:
            if loader == self._longest_loader:
                loaders.append(cast(Iterator[T], loader))
            else:
                loaders.append(MultitaskLoader.cycle(loader))

        for batch in zip(*loaders):
            yield batch
