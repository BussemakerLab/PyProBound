"""Typed helper functions."""

from typing import Hashable, Iterable, Sequence, overload

import torch
import torch.mps
import torch.nn.functional as F
from scipy.sparse import csc_array
from torch import Tensor


@overload
def ceil_div(dividend: int, divisor: int) -> int: ...


@overload
def ceil_div(dividend: Tensor, divisor: int) -> Tensor: ...


@overload
def ceil_div(dividend: int, divisor: Tensor) -> Tensor: ...


@overload
def ceil_div(dividend: Tensor, divisor: Tensor) -> Tensor: ...


def ceil_div(dividend: int | Tensor, divisor: int | Tensor) -> int | Tensor:
    """Typed ceiling division."""
    return -(-dividend // divisor)


@overload
def floor_div(dividend: int, divisor: int) -> int: ...


@overload
def floor_div(dividend: Tensor, divisor: int) -> Tensor: ...


@overload
def floor_div(dividend: int, divisor: Tensor) -> Tensor: ...


@overload
def floor_div(dividend: Tensor, divisor: Tensor) -> Tensor: ...


def floor_div(dividend: int | Tensor, divisor: int | Tensor) -> int | Tensor:
    """Typed floor division."""
    return dividend // divisor


def log1mexp(tensor: Tensor, /, eps: float = 1e-8) -> Tensor:
    r"""Computes the element-wise log1mexp in a numerically stable way.

    .. math::
        \log \left( 1 - e^{-x} \right)

    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
    """
    tensor = torch.where(torch.abs(tensor) < eps, eps, torch.abs(tensor))
    return torch.where(
        tensor > 0.693,
        torch.log1p(-torch.exp(-tensor)),
        torch.log(-torch.expm1(-tensor)),
    )


def logsigmoid(tensor: Tensor, /, threshold: int = 20) -> Tensor:
    r"""Computes the element-wise logsigmoid using softplus for stability.

    .. math::
        \log \frac{1}{1 + e^{-x}}
    """
    return -F.softplus(-tensor, threshold=threshold)


def betaln(z_1: Tensor, z_2: Tensor) -> Tensor:
    r"""Computes the natural logarithm of the beta function.

    .. math::
        \log \frac{\Gamma(z_1) \Gamma(z_2)}{\Gamma(z_1 + z_2)}
    """
    return torch.lgamma(z_1) + torch.lgamma(z_2) - torch.lgamma(z_1 + z_2)


def avg_pool1d(tensor: Tensor, kernel: int = 1) -> Tensor:
    """Average pooling along the first dimension."""
    if kernel <= 1:
        return tensor
    dims = tensor.dim()
    if dims == 0:
        raise ValueError("No dimensions to pool over")
    if dims == 1:
        return F.avg_pool1d(tensor.unsqueeze(0).unsqueeze(0), kernel).flatten()
    if dims == 2:
        return F.avg_pool1d(tensor.T.unsqueeze(1), kernel).squeeze(1).T
    return F.avg_pool1d(tensor.transpose(0, -1), kernel).transpose(0, -1)


def get_split_size(
    max_embedding_size: int, max_split: int, device: str | torch.device
) -> int:
    """Calculates the minibatch needed to avoid GPU limits on tensor sizes.

    Args:
        max_embedding_size: The maximum number of bytes needed to encode a
            sequence.
        max_split: Maximum number of sequences scored at a time
            (lower values reduce memory but increase computation time).
        device: The current device of the model.
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps":
        return min(max_split, 65_535, 2**27 // max_embedding_size)
    return max_split


def get_ordinal(integer: int) -> str:
    """Converts an integer to a string with an ordinal suffix."""
    if integer % 100 in (11, 12, 13):
        return f"{integer}th"
    match integer % 10:
        case 1:
            return f"{integer}st"
        case 2:
            return f"{integer}nd"
        case 3:
            return f"{integer}rd"
        case _:
            return f"{integer}th"


def clear_cache() -> None:
    """Calls the empty_cache() function matching the available GPU backends."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def count_kmers(
    sequences: Iterable[Sequence[Hashable]],
    kmer_length: int = 3,
    vocabulary: dict[Sequence[Hashable], int] | None = None,
) -> tuple[csc_array, dict[Sequence[Hashable], int]]:
    """Returns a sparse count matrix of k-mers in a list of sequences.

    Args:
        sequences: The sequences to count the k-mers in.
        kmer_length: The k-mer length to be counted.
        vocabulary: Mapping of k-mers to indices.

    Returns:
        A tuple (matrix, vocabulary), where matrix is a sparse CSC matrix of
        the count of each k-mer in each sequence, and vocabulary is the mapping
        of k-mers to their respective indices in the matrix.
    """
    if vocabulary is None:
        vocabulary = {}
    data: list[int] = []
    indices: list[int] = []
    indptr: list[int] = [0]
    num_seqs = 0
    for seq in sequences:
        num_seqs += 1
        count: dict[int, int] = {}
        for view in range(0, len(seq) - kmer_length + 1):
            kmer = seq[view : view + kmer_length]

            # Get key from vocabulary
            if kmer not in vocabulary:
                key = len(vocabulary)
                vocabulary[kmer] = key
            else:
                key = vocabulary[kmer]

            # Running sum of kmers in sequence
            if key not in count:
                count[key] = 1
            else:
                count[key] += 1

        # Add to csc initializers
        indptr.append(indptr[-1] + len(count))
        data.extend(count.values())
        indices.extend(count.keys())

    sparse = csc_array(
        (data, indices, indptr), shape=(len(vocabulary), num_seqs)
    )
    return sparse, vocabulary
