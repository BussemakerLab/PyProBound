"""Collection of small calculation functions"""

from typing import cast, overload

import torch
import torch.mps
from torch import Tensor


@overload
def ceil_div(dividend: int, divisor: int) -> int:
    ...


@overload
def ceil_div(dividend: Tensor, divisor: int) -> Tensor:
    ...


@overload
def ceil_div(dividend: int, divisor: Tensor) -> Tensor:
    ...


@overload
def ceil_div(dividend: Tensor, divisor: Tensor) -> Tensor:
    ...


def ceil_div(dividend: int | Tensor, divisor: int | Tensor) -> int | Tensor:
    return -(-dividend // divisor)


@overload
def floor_div(dividend: int, divisor: int) -> int:
    ...


@overload
def floor_div(dividend: Tensor, divisor: int) -> Tensor:
    ...


@overload
def floor_div(dividend: int, divisor: Tensor) -> Tensor:
    ...


@overload
def floor_div(dividend: Tensor, divisor: Tensor) -> Tensor:
    ...


def floor_div(dividend: int | Tensor, divisor: int | Tensor) -> int | Tensor:
    return dividend // divisor


def log1mexp(tensor: Tensor, /, eps: float = 1e-8) -> Tensor:
    """Numerically stable log(1 - exp(-abs(tensor)))"""
    # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    tensor = torch.where(torch.abs(tensor) < eps, eps, torch.abs(tensor))
    return torch.where(
        tensor > 0.693,
        torch.log1p(-torch.exp(-tensor)),
        torch.log(-torch.expm1(-tensor)),
    )


def logsigmoid(tensor: Tensor, /, threshold: float = 20) -> Tensor:
    return cast(
        Tensor, -torch.nn.functional.softplus(-tensor, threshold=threshold)
    )


def betaln(a: Tensor, b: Tensor) -> Tensor:
    return cast(
        Tensor,
        (
            torch.special.gammaln(a)
            + torch.special.gammaln(b)
            - torch.special.gammaln(a + b)
        ),
    )


def avg_pool1d(tensor: Tensor, kernel: int = 1) -> Tensor:
    """Average pooling along the first dimension"""
    if kernel <= 1:
        return tensor
    dims = tensor.dim()
    if dims == 0:
        raise ValueError("No dimensions to pool over")
    if dims == 1:
        return torch.nn.functional.avg_pool1d(
            tensor.unsqueeze(0).unsqueeze(0), kernel
        ).flatten()
    if dims == 2:
        return (
            torch.nn.functional.avg_pool1d(tensor.T.unsqueeze(1), kernel)
            .squeeze(1)
            .T
        )
    return torch.nn.functional.avg_pool1d(
        tensor.transpose(0, -1), kernel
    ).transpose(0, -1)


def get_split_size(
    max_embedding_size: int, max_split: int, device: str | torch.device
) -> int:
    """Maximum batch dimension
    See https://github.com/pytorch/pytorch/issues/96225
    See https://github.com/pytorch/pytorch/issues/96716
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps":
        return min(max_split, 65_535, (2**31 // max_embedding_size) - 1)
    return max_split


def get_ordinal(n: int) -> str:
    """Convert int to str with ordinal suffix"""
    if n % 100 in (11, 12, 13):
        return f"{n}th"
    match n % 10:
        case 1:
            return f"{n}st"
        case 2:
            return f"{n}nd"
        case 3:
            return f"{n}rd"
        case _:
            return f"{n}th"


def clear_cache() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
