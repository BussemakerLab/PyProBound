"""Module for loading and scoring count tables and sequences.

Members are explicitly re-exported in pyprobound.

A count table consists of sequences and their corresponding counts within each
selection round. For example, a three-round SELEX table might be:
AAAA    0   0   2
ACGA    2   1   0
CGAA    0   1   5
TCAG    1   0   0

A table might also contain flanking sequences on the left and right, which are
prepended and appended, respectively, to every sequence in the table.
"""

import abc
import functools
import gzip
import itertools
import os
import warnings
from collections.abc import Callable, Iterable, Iterator, Sized
from typing import Any, Generic, NamedTuple, Protocol, TypeVar, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler, SubsetRandomSampler
from typing_extensions import override

from . import __precision__
from .alphabets import Alphabet
from .base import Transform
from .utils import get_split_size

T = TypeVar("T")


class CountBatch(Protocol):  # pylint: disable=too-few-public-methods
    r"""A protocol for a set of rows from a count table.

    Attributes:
        seqs: A sequence tensor of shape
            :math:`(\text{minibatch},\text{length})` or
            :math:`(\text{minibatch},\text{in_channels},\text{length})`.
        target: A count tensor of shape
            :math:`(\text{minibatch},\text{rounds})`.
    """

    seqs: Tensor
    target: Tensor


class CountBatchTuple(NamedTuple):
    r"""A NamedTuple for a set of rows from a count table.

    Attributes:
        seqs: A sequence tensor of shape
            :math:`(\text{minibatch},\text{length})` or
            :math:`(\text{minibatch},\text{in_channels},\text{length})`.
        target: A count tensor of shape
            :math:`(\text{minibatch},\text{rounds})`.
    """

    seqs: Tensor
    target: Tensor


def score(
    module: Transform,
    batch: CountBatch,
    fun: str = "forward",
    max_split: int | None = None,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Scores a batch using a chosen function, automatically managing devices.

    Args:
        module: The Transform used for scoring.
        batch: The CountBatch containing the sequences and counts to be scored.
        fun: The name of the function taken from the module for scoring.
        max_split: Maximum number of sequences scored at a time.
        kwargs: Any keyword arguments passed to the function.

    Returns:
        A tuple of the observed counts and predicted scores, both on CPU.
    """
    method: Callable[[Tensor], Tensor] = getattr(module, fun)
    for p in module.parameters():
        device = p.device
        break
    split_size = get_split_size(
        module.max_embedding_size(),
        (
            len(batch.seqs)
            if max_split is None
            else min(max_split, len(batch.seqs))
        ),
        device,
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
    """Loads tab-delimited count tables into columns on a Pandas dataframe.

    The input count tables are assumed to have a sequence field and a series
    of count fields, all separated by a tab character. The first line is
    automatically skipped if it does not contain a tab character.

    Args:
        paths: The paths to each count table to be merged into a dataframe.
        total_count: The total number of counts to be sampled from each column.
        random_state: A seed used to make the output reproducible if
            `total_count` is specified.

    Returns:
        An integer Pandas dataframe with each column representing a sequencing
        round. Sequences are stored in the index of the dataframe.
    """

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
    """Randomly samples from a dataframe evenly by enrichment.

    To make validation or test splits representative of the training data, bin
    sequences by their overall enrichment and sample evenly within each bin.

    Args:
        dataframe: The input dataframe to be sampled from.
        frac: The proportion of reads to be sampled from the dataframe.
        random_state: A seed used to make the output reproducible.
        n_bin: The bin size used to sample sequences from.

    Returns:
        A tuple of two dataframes, the first containing `frac` of the original
        dataframe, the second containing `1 - frac` of the original dataframe.
    """
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
    """Randomly samples `n_counts` total counts each column of a dataframe.

    Args:
        dataframe: The dataframe to sample counts from.
        n_counts: The total number of counts to be included in the output.
        random_state: A seed used to make the output reproducible.

    Returns:
        A dataframe with approximately `n_counts` total counts.
    """
    generator = np.random.default_rng(random_state)
    for i in range(dataframe.shape[1]):
        total = dataframe.iloc[:, i].sum()
        dataframe.iloc[:, i] = generator.binomial(
            dataframe.iloc[:, i], min(1, n_counts / total)
        )
    return dataframe[dataframe.sum(axis=1) != 0]


class Table(Dataset[T], Generic[T], Sized, abc.ABC):
    """A generic tensor encoding of a table, should implement flank management.

    Attributes:
        left_flank (str): The prepended sequence.
        right_flank (str): The appended sequence.
        left_flank_length (int): The scored length of the left flank.
        right_flank_length (int): The scored length of the right flank.
    """

    def __init__(
        self,
        left_flank: str = "",
        right_flank: str = "",
        left_flank_length: int = 0,
        right_flank_length: int = 0,
    ) -> None:
        r"""Initializes the table.

        Args:
            left_flank (str): The prepended sequence.
            right_flank (str): The appended sequence.
            left_flank_length (int): The scored length of the left flank.
            right_flank_length (int): The scored length of the right flank.
        """
        self._left_flank = left_flank
        self._right_flank = right_flank
        self._left_flank_length = 0
        self._right_flank_length = 0
        self.set_flank_length(left=left_flank_length, right=right_flank_length)

    @property
    def left_flank(self) -> str:
        """The prepended sequence."""
        return self._left_flank

    @property
    def right_flank(self) -> str:
        """The appended sequence."""
        return self._right_flank

    @property
    def left_flank_length(self) -> int:
        """The scored length of the left flank."""
        return self._left_flank_length

    @property
    def right_flank_length(self) -> int:
        """The scored length of the right flank."""
        return self._right_flank_length

    @property
    @abc.abstractmethod
    def input_shape(self) -> int:
        """The number of elements in the length dimension."""

    @property
    @abc.abstractmethod
    def min_read_length(self) -> int:
        """The minimum number of finite elements in the length dimension."""

    @property
    @abc.abstractmethod
    def max_read_length(self) -> int:
        """The maximum number of finite elements in the length dimension."""

    @abc.abstractmethod
    def get_setup_string(self) -> str:
        """A description used when printing the output of an optimizer."""

    @abc.abstractmethod
    def set_flank_length(self, left: int = 0, right: int = 0) -> None:
        """Updates the length of flanks included in sequences.

        Args:
            left: The new length of the prepended sequence.
            right: The new length of the appended sequence.
        """


class CountTable(Table[CountBatch]):
    r"""A tensor encoding of a count table with flank management.

    Attributes:
        seqs (Tensor): A sequence tensor of shape
            :math:`(\text{minibatch},\text{length})` or
            :math:`(\text{minibatch},\text{in_channels},\text{length})`.
        target (Tensor): A count tensor of shape
            :math:`(\text{minibatch},\text{rounds})`.
        left_flank (str): The prepended sequence.
        right_flank (str): The appended sequence.
        left_flank_length (int): The scored length of the left flank.
        right_flank_length (int): The scored length of the right flank.
        counts_per_round (Tensor): The number of probes in each round of the
            count table, as a count tensor of shape :math:`(\text{rounds})`.
    """

    def __init__(
        self,
        dataframe: DataFrame,
        alphabet: Alphabet,
        transliterate: dict[str, str] | None = None,
        left_flank: str = "",
        right_flank: str = "",
        left_flank_length: int = 0,
        right_flank_length: int = 0,
        max_left_flank_length: int | None = None,
        max_right_flank_length: int | None = None,
        wildcard_pad: bool = False,
        min_variable_length: int | None = None,
        max_variable_length: int | None = None,
    ) -> None:
        r"""Initializes the count table.

        Args:
            dataframe: The dataframe used to initialize the count table.
            alphabet: The alphabet used to encode sequences into tensors.
            transliterate: A mapping of strings to be replaced before encoding.
            left_flank (str): The prepended sequence.
            right_flank (str): The appended sequence.
            left_flank_length (int): The scored length of the left flank.
            right_flank_length (int): The scored length of the right flank.
            max_left_flank_length: The maximum allowed length of the prepended
                sequence.
            max_right_flank_length: The maximum allowed length of the appended
                sequence.
            wildcard_pad: Whether to append a wildcard character
                (ex. N for DNA) to all sequences to make them the same length.
            min_variable_length: The minimum possible length of the sequences
                (needed if using train/test splits on variable length data).
            max_variable_length: The maximum possible length of the sequences
                (needed if using train/test splits on variable length data).
        """

        # Check dataframe
        if any((dataframe <= 0).all(axis=1)):
            warnings.warn(
                "Some sequences do not have a positive count in any round"
            )

        # Transliterate if necessary
        if transliterate is not None:
            for pattern, replace in transliterate.items():
                dataframe.index = dataframe.index.str.replace(pattern, replace)
                left_flank = left_flank.replace(pattern, replace)
                right_flank = right_flank.replace(pattern, replace)

        # Instance attributes
        self.alphabet = alphabet
        if max_left_flank_length is None:
            max_left_flank_length = len(left_flank)
        if max_right_flank_length is None:
            max_right_flank_length = len(right_flank)
        self.max_left_flank_length = max_left_flank_length
        self.max_right_flank_length = max_right_flank_length
        self.wildcard_padded = wildcard_pad

        # Get dataframe data
        self._padding_value = alphabet.get_index["*" if wildcard_pad else " "]
        self.target = torch.tensor(dataframe.values, dtype=__precision__)
        self.seqs = torch.nn.utils.rnn.pad_sequence(
            [alphabet.translate(seq) for seq in dataframe.index],
            batch_first=True,
            padding_value=self._padding_value,
        )

        # Store variable lengths
        self.variable_lengths = torch.sum(
            self.seqs != self.alphabet.neginf_pad, dim=1
        ).unsqueeze(-1)
        curr_min_variable_length = self.variable_lengths.min().item()
        curr_max_variable_length = self.variable_lengths.max().item()
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
        self._min_variable_length = min_variable_length
        self._max_variable_length = max_variable_length
        self.seqs = F.pad(
            self.seqs,
            (0, self._max_variable_length - self.input_shape),
            value=self._padding_value,
        ).contiguous()

        # Get number of probes per round
        self.counts_per_round = torch.sum(self.target, dim=0)

        # Set flank length
        super().__init__(
            left_flank=left_flank,
            right_flank=right_flank,
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
        if self.wildcard_padded:
            return self.max_read_length
        return (
            self._min_variable_length
            + self._left_flank_length
            + self._right_flank_length
        )

    @override
    @property
    def max_read_length(self) -> int:
        return (
            self._max_variable_length
            + self._left_flank_length
            + self._right_flank_length
        )

    @override
    def get_setup_string(self) -> str:
        return "\n".join(
            [
                f"\t\tMaximum Variable Length: {self._max_variable_length}",
                f"\t\tLeft Flank Length: {self._left_flank_length}",
                f"\t\tRight Flank Length: {self._right_flank_length}",
            ]
        )

    @override
    def set_flank_length(self, left: int = 0, right: int = 0) -> None:
        # Check input
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
        if (
            self._left_flank_length == left
            and self._right_flank_length == right
        ):
            return

        # Trim new flanks to desired lengths
        old_left_flank = self.left_flank[
            len(self.left_flank) - self._left_flank_length :
        ]
        left_flank = self.left_flank.rjust(left, "*")[
            len(self.left_flank) - left :
        ]
        old_left_flank_tr = self.alphabet.translate(old_left_flank)
        left_flank_tr = self.alphabet.translate(left_flank)
        old_right_flank = self.right_flank[: self._right_flank_length]
        right_flank = self.right_flank.ljust(right, "*")[:right]
        right_flank_tr = self.alphabet.translate(right_flank)
        old_right_flank_tr = self.alphabet.translate(old_right_flank)

        # Update left flank
        if len(left_flank_tr) < len(old_left_flank_tr):
            # Trimming left flank
            self.seqs = self.seqs[
                :, len(old_left_flank_tr) - len(left_flank_tr) :
            ]
        elif left > self._left_flank_length:
            # Extending left flank
            left_update = left_flank_tr[
                : len(left_flank_tr) - len(old_left_flank_tr)
            ]
            self.seqs = torch.hstack(
                (left_update.expand(len(self), -1), self.seqs)
            )
        self._left_flank_length = left

        # Update right flank
        if len(right_flank_tr) < len(old_right_flank_tr):
            # Trimming right flank
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
            # Extending right flank
            diff = len(right_flank_tr) - len(old_right_flank_tr)
            self.seqs = F.pad(self.seqs, (0, diff), value=self._padding_value)
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
        self._right_flank_length = right

        # Make contiguous
        self.seqs = self.seqs.contiguous()

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
        self._min_variable_length = cast(
            int, self.variable_lengths.min().item()
        )
        self._max_variable_length = cast(
            int, self.variable_lengths.max().item()
        )
        self.counts_per_round = torch.sum(self.target, dim=0)

    @override
    def __getitem__(self, idx: int) -> CountBatch:
        return cast(
            CountBatch, CountBatchTuple(self.seqs[idx], self.target[idx])
        )

    @override
    def __len__(self) -> int:
        return len(self.seqs)


class EvenSampler(Sampler[int]):
    """Evenly sample across the range of indices"""

    def __init__(self, data_source: Sized, n_bin: int = 128) -> None:
        super().__init__(data_source=None)
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


class MultitaskLoader(Generic[T]):  # pylint: disable=too-few-public-methods
    """Combines multiple dataloaders for multitask learning.

    Attributes:
        dataloaders: The dataloaders iterated through together.
    """

    def __init__(self, dataloaders: Iterable[DataLoader[T]]) -> None:
        self.dataloaders = dataloaders
        self._longest_loader = max(self.dataloaders, key=len)

    @staticmethod
    def _cycle(iterable: Iterable[T]) -> Iterator[T]:
        """Loop infinitely through an iterable."""
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    def __iter__(self) -> Iterator[tuple[T, ...]]:
        if all(len(loader) == 1 for loader in self.dataloaders):
            yield tuple(cast(T, loader.dataset) for loader in self.dataloaders)
            return

        loaders: list[Iterator[T]] = []
        for loader in self.dataloaders:
            if loader == self._longest_loader:
                loaders.append(cast(Iterator[T], loader))
            else:
                loaders.append(MultitaskLoader._cycle(loader))

        for batch in zip(*loaders):
            yield batch
