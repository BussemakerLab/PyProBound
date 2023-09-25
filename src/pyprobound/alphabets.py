"""Alphabets for encoding sequences into tensors."""
import itertools
from collections.abc import Iterable, Iterator, Mapping
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from . import __precision__

_iupac = {
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
}


class Alphabet:
    """Stores the alphabet encoding of sequences into tensors.

    Assumes that the reverse complement mapping is `dict(zip(a, reversed(a)))`.
    Three sequence characters are reserved: ' ' is -infinity (not scored),
    '*' is an uninformative prior over channels, and '-' is zero.

    Attributes:
        alphabet (tuple[str]): The monomers of the alphabet.
        get_index (dict[str, int]): A mapping of monomers in the alphabet to
            indices in the embedding matrix.
        get_encoding (dict[str, tuple[int,...]]): A mapping of monomers to
            tuples of indices in the embedding matrix; for example, '*' maps to
            all indices available in the embedding.
        get_inv_encoding (dict[tuple[int,...], str]): Inverse of
            `get_encoding`.
    """

    def __init__(
        self,
        alphabet: Iterable[str],
        complement: bool = False,
        monomer_length: int = 1,
        encoding: Mapping[str, Iterable[str]] | None = None,
        color_scheme: str | dict[str, str | list[float]] | None = None,
    ) -> None:
        """Initializes the alphabet.

        Args:
            alphabet: The monomers of the alphabet.
            complement: Whether to take the reverse order of the alphabet as
                the complement encoding - for example, the complement of
                `['A','C','G','T']` would be assumed to be `['T','G','C','A']`.
            monomer_length: The length of elements in the alphabet.
            encoding: A mapping of monomers to a degenerate list of monomers -
                for example, 'N' maps to `['A','C','G','T']`.
            color_scheme: Passed to Logomaker.Logo.
        """
        # Store attributes
        self.alphabet = tuple(alphabet)
        self.complement = complement
        self.color_scheme = color_scheme
        self.monomer_length = monomer_length

        # Assemble get_index (maps monomers to indices in embedding matrix)
        self.get_index: dict[str, int] = {
            val: idx for idx, val in enumerate(self.alphabet)
        }
        self.neginf_pad = len(self.alphabet)
        self.get_index[" " * self.monomer_length] = self.neginf_pad

        # Assemble get_encoding (maps monomers to groups of indices ex. iupac)
        self.encoding: list[str] = []
        self.get_encoding: dict[str, tuple[int, ...]] = {}
        self.get_inv_encoding: dict[tuple[int, ...], str] = {}
        for val, idx in self.get_index.items():
            encoded: tuple[int, ...] = (idx,)
            self.encoding.append(val)
            self.get_encoding[val] = encoded
            self.get_inv_encoding[encoded] = val

        # Add encoding
        extended_encoding: dict[str, Iterable[str]] = {
            "*" * self.monomer_length: self.alphabet,
            "-" * self.monomer_length: [],
        }
        if encoding is not None:
            extended_encoding.update(encoding)
        for code, mapping in extended_encoding.items():
            encoded = tuple(self.get_index[i] for i in mapping)
            self.encoding.append(code)
            alt_code = self.get_inv_encoding.get(encoded, None)
            self.get_encoding[code] = encoded
            if alt_code is not None:
                self.get_index[code] = self.get_index[alt_code]
            else:
                self.get_index[code] = len(self.get_index)
                self.get_inv_encoding[encoded] = code

        # Create embedding matrix
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
                    encoded = self.get_encoding[code]
                    embedding_list.append(
                        (
                            F.one_hot(
                                torch.tensor(encoded),
                                num_classes=len(self.alphabet),
                            ).sum(0)
                            / len(encoded)
                        ).to(dtype=__precision__)
                    )
                else:
                    embedding_list.append(
                        torch.zeros(len(self.alphabet), dtype=__precision__)
                    )
        embedding_weight = torch.stack(embedding_list)
        self.embedding: torch.nn.Embedding
        self.embedding = torch.nn.Embedding.from_pretrained(embedding_weight)  # type: ignore[no-untyped-call]
        pairwise_embedding_weight = torch.stack(
            [
                (
                    embedding_weight[i].unsqueeze(1)
                    * embedding_weight[j].unsqueeze(0)
                ).flatten()
                for i in range(embedding_weight.shape[0])
                for j in range(embedding_weight.shape[0])
            ]
        )
        self.pairwise_embedding: torch.nn.Embedding
        self.pairwise_embedding = torch.nn.Embedding.from_pretrained(  # type: ignore[no-untyped-call]
            pairwise_embedding_weight
        )

    def translate(self, sequence: str) -> Tensor:
        r"""Translates a sequence into a tensor.

        Args:
            sequence: A string sequence of length :math:`\text{length}`.

        Returns:
            A dense representation of the sequence as an integer tensor of
            shape :math:`(\text{length},)`.
        """

        def _chunk(seq: str) -> Iterator[str]:
            """Yield successive n-sized chunks from seq"""
            for i in range(0, len(seq), self.monomer_length):
                yield seq[i : i + self.monomer_length]

        return torch.tensor(
            [self.get_index[char] for char in _chunk(sequence)],
            dtype=torch.int64,
        )

    def embed(self, seqs: Tensor) -> Tensor:
        r"""Embeds sequences from a dense to a one-hot representation.

        Args:
            seqs: A dense representation of sequences as an integer tensor of
                shape :math:`(\text{minibatch},\text{length})`.

        Returns:
            A one-hot embedding of the sequences as a float tensor of shape
            :math:`(\text{minibatch},\text{channels},\text{length})`.
        """
        return cast(
            Tensor, self.embedding.to(seqs.device)(seqs).transpose(1, 2)
        )

    def pairwise_embed(self, seqs: Tensor, dist: int) -> Tensor:
        r"""Embeds sequences into a one-hot pairwise representation.

        Args:
            seqs: A dense representation of sequences as an integer tensor of
                shape :math:`(\text{minibatch},\text{length})`.
            dist: The pairwise distance between two monomers.

        Returns:
            A one-hot embedding of the sequences as a float tensor of shape
            :math:`(\text{minibatch}, \text{channels}^2, \text{length}
            - \text{dist})`. Each position `i` in the last dimension contains
            the product of the embedding of `i` and `i+dist`.
        """
        if dist < 1:
            raise ValueError(
                f"dist must be greater than 0, found {dist} instead"
            )
        return cast(
            Tensor,
            self.pairwise_embedding.to(seqs.device)(
                seqs[:, :-dist] * len(self.embedding.weight) + seqs[:, dist:]
            ).transpose(1, 2),
        )


class DNA(Alphabet):
    """Stores the DNA encoding of sequences into tensors.

    Three sequence characters are reserved: ' ' is -infinity (not scored),
    '*' is an uninformative prior over channels, and '-' is zero.

    Attributes:
        alphabet (tuple[str]): ('A', 'C', 'G', 'T').
        get_index (dict[str, int]): A mapping of monomers in the alphabet to
            indices in the embedding matrix.
        get_encoding (dict[str, tuple[int,...]]): IUPAC encoding of monomers to
            tuples of indices in the embedding matrix; for example, 'N' maps to
            (0, 1, 2, 3).
    """

    def __init__(self) -> None:
        """Initializes the DNA alphabet."""
        super().__init__(
            alphabet=["A", "C", "G", "T"],
            complement=True,
            color_scheme="classic",
            encoding=_iupac,
        )


class RNA(Alphabet):
    """Stores the RNA encoding of sequences into tensors.

    Three sequence characters are reserved: ' ' is -infinity (not scored),
    '*' is an uninformative prior over channels, and '-' is zero.

    Attributes:
        alphabet (tuple[str]): ('A', 'C', 'G', 'U').
        get_index (dict[str, int]): A mapping of monomers in the alphabet to
            indices in the embedding matrix.
        get_encoding (dict[str, tuple[int,...]]): IUPAC encoding of monomers to
            tuples of indices in the embedding matrix; for example, 'N' maps to
            (0, 1, 2, 3).
    """

    def __init__(self) -> None:
        """Initializes the alphabet."""
        super().__init__(
            alphabet=["A", "C", "G", "U"],
            complement=False,
            color_scheme="classic",
            encoding={
                key: ["U" if i == "T" else i for i in val]
                for key, val in _iupac.items()
            },
        )


class Codon(Alphabet):
    r"""Stores the codon encoding of sequences into tensors.

    Three sequence characters are reserved: '   ' is -infinity (not scored),
    '***' is an uninformative prior over channels, and '---' is zero.

    Attributes:
        alphabet (tuple[str]): All :math:`_{4}P_{3}` permutations of the DNA
            alphabet.
        get_index (dict[str, int]): A mapping of monomers in the alphabet to
            indices in the embedding matrix.
        get_encoding (dict[str, tuple[int,...]]): IUPAC encoding of monomers to
            tuples of indices in the embedding matrix; for example, '***' maps
            to (0, 1, ..., 63).
    """

    def __init__(self) -> None:
        """Initializes the alphabet."""
        super().__init__(
            alphabet=[
                "".join(i)
                for i in itertools.product(["A", "C", "G", "T"], repeat=3)
            ],
            monomer_length=3,
            complement=False,
            color_scheme="classic",
            encoding={
                "".join([j[0] for j in i]): [
                    "".join(k) for k in itertools.product(*(j[1] for j in i))
                ]
                for i in itertools.product(_iupac.items(), repeat=3)
            },
        )


class Protein(Alphabet):
    """Stores the protein encoding of sequences into tensors.

    Three sequence characters are reserved: ' ' is -infinity (not scored),
    '*' is an uninformative prior over channels, and '-' is zero.

    Attributes:
        alphabet (tuple[str]): All 20 one-letter amino acid codes.
        get_index (dict[str, int]): A mapping of monomers in the alphabet to
            indices in the embedding matrix.
        get_encoding (dict[str, tuple[int,...]]): IUPAC encoding of monomers to
            tuples of indices in the embedding matrix; for example, 'X' maps
            to (0, 1, ..., 19).
    """

    def __init__(self) -> None:
        """Initializes the alphabet."""
        super().__init__(
            # fmt: off
            alphabet=["A", "C", "D", "E",
                      "F", "G", "H", "I",
                      "K", "L", "M", "N",
                      "P", "Q", "R", "S",
                      "T", "V", "W", "Y"],
            complement=False,
            color_scheme="chemistry",
            encoding={
                "B": ["D", "N"],
                "Z": ["E", "Q"],
                "J": ["I", "L"],
                "X": ["A", "C", "D", "E",
                      "F", "G", "H", "I",
                      "K", "L", "M", "N",
                      "P", "Q", "R", "S",
                      "T", "V", "W", "Y"],
            },
            # fmt: on
        )
