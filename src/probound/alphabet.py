"""Module for Alphabet class to avoid circular imports"""
import itertools
from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

from . import __precision__


class Alphabet:
    """Stores alphabet encoding, assumes alphabet[::-1] is complement

    " " = -inf, "*" = even prior, "-" = zero
    """

    def __init__(
        self,
        alphabet: Iterable[str],
        complement: bool = False,
        monomer_length: int = 1,
        encoding: dict[str, Iterable[str]] | None = None,
        color_scheme: str | dict[str, str | list[float]] | None = None,
    ) -> None:
        # store attributes
        self.alphabet = tuple(alphabet)
        self.complement = complement
        self.color_scheme = color_scheme
        self.monomer_length = monomer_length

        # assemble get_index (maps monomers to indices in embedding matrix)
        self.get_index: dict[str, int] = {
            val: idx for idx, val in enumerate(self.alphabet)
        }
        self.neginf_pad = len(self.alphabet)
        self.get_index[" " * self.monomer_length] = self.neginf_pad

        # assemble get_encoding (maps monomers to groups of indices ex. iupac)
        self.encoding: list[str] = []
        self.get_encoding: dict[str, tuple[int, ...]] = {}
        for val, idx in self.get_index.items():
            self.encoding.append(val)
            self.get_encoding[val] = (idx,)

        # add encoding
        extended_encoding: dict[str, Iterable[str]] = {
            "*" * self.monomer_length: self.alphabet,
            "-" * self.monomer_length: [],
        }
        if encoding is not None:
            extended_encoding.update(encoding)
        for code, mapping in extended_encoding.items():
            self.encoding.append(code)
            self.get_index[code] = len(self.get_index)
            self.get_encoding[code] = tuple(self.get_index[i] for i in mapping)

        # create embedding matrix
        embedding_list: list[Tensor] = []
        for code in self.encoding:
            if code == self.encoding[self.neginf_pad]:
                embedding_list.append(
                    torch.full(
                        (len(self.alphabet),),
                        float("-inf"),
                        dtype=__precision__,
                    )
                )
            else:
                if len(self.get_encoding[code]) > 0:
                    embedding_list.append(
                        F.one_hot(
                            torch.tensor(self.get_encoding[code]),
                            num_classes=len(self.alphabet),
                        )
                        .sum(0)
                        .to(dtype=__precision__)
                    )
                else:
                    embedding_list.append(
                        torch.zeros(len(self.alphabet), dtype=__precision__)
                    )
        embedding_weight = torch.stack(embedding_list)
        self.embedding: torch.nn.Embedding
        self.embedding = torch.nn.Embedding.from_pretrained(embedding_weight)  # type: ignore[no-untyped-call]
        interaction_embedding_weight = torch.stack(
            [
                (
                    embedding_weight[i].unsqueeze(1)
                    * embedding_weight[j].unsqueeze(0)
                ).flatten()
                for i in range(embedding_weight.shape[0])
                for j in range(embedding_weight.shape[0])
            ]
        )
        self.interaction_embedding: torch.nn.Embedding
        self.interaction_embedding = torch.nn.Embedding.from_pretrained(  # type: ignore[no-untyped-call]
            interaction_embedding_weight
        )

    def translate(self, sequence: str) -> Tensor:
        """Translates a sequence into a tensor"""

        def _chunk(seq: str) -> Iterator[str]:
            """Yield successive n-sized chunks from seq"""
            for i in range(0, len(seq), self.monomer_length):
                yield seq[i : i + self.monomer_length]

        return torch.tensor(
            [self.get_index[char] for char in _chunk(sequence)],
            dtype=torch.int64,
        )


class DNA(Alphabet):
    def __init__(self) -> None:
        super().__init__(
            alphabet=["A", "C", "G", "T"],
            complement=True,
            color_scheme="classic",
            encoding={
                "M": ["A", "C"],
                "R": ["A", "G"],
                "W": ["A", "T"],
                "S": ["C", "G"],
                "Y": ["C", "T"],
                "K": ["G", "T"],
                "V": ["A", "C", "G"],
                "H": ["A", "C", "T"],
                "D": ["A", "G", "T"],
                "B": ["C", "G", "T"],
                "N": ["A", "C", "G", "T"],
            },
        )


class RNA(Alphabet):
    def __init__(self) -> None:
        super().__init__(
            alphabet=["A", "C", "G", "U"],
            complement=False,
            color_scheme="classic",
            encoding={
                "M": ["A", "C"],
                "R": ["A", "G"],
                "W": ["A", "U"],
                "S": ["C", "G"],
                "Y": ["C", "U"],
                "K": ["G", "U"],
                "V": ["A", "C", "G"],
                "H": ["A", "C", "U"],
                "D": ["A", "G", "U"],
                "B": ["C", "G", "U"],
                "N": ["A", "C", "G", "U"],
            },
        )


class Codon(Alphabet):
    def __init__(self) -> None:
        super().__init__(
            alphabet=[
                "".join(i)
                for i in itertools.product(["A", "C", "G", "T"], repeat=3)
            ],
            monomer_length=3,
            complement=False,
            color_scheme="classic",
        )


class Protein(Alphabet):
    def __init__(self) -> None:
        super().__init__(
            # fmt: off
            alphabet=["A", "C", "D", "E",
                      "F", "G", "H", "I",
                      "K", "L", "M", "N",
                      "P", "Q", "R", "S",
                      "T", "V", "W", "Y"],
            # fmt: on
            complement=False,
            color_scheme="chemistry",
        )
