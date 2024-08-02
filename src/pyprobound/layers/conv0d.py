"""Nonspecific binding layer.

Members are explicitly re-exported in pyprobound.layers.

Non-specific binding is equivalent to a traditional weight matrix with all its
values constrained to be identical to each other, so the only thing that can
change the output is the length of the sequence, not its identity. It sets a
background enrichment level from which a sequence-specific enrichment is
optimized. If it is included in an aggregate, it should be listed first.
"""

from typing import Any, Literal, TypeVar, overload

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self, override

from .. import __precision__
from ..alphabets import Alphabet
from ..base import BindingOptim, Call, Step
from ..table import Table
from .layer import Layer, LayerSpec

T = TypeVar("T", int, Tensor)


class NonSpecific(LayerSpec):
    """Non-specific factor, equivalent to a PSAM of size 1 and equal betas."""

    def __init__(
        self, alphabet: Alphabet, ignore_length: bool = False, name: str = ""
    ) -> None:
        """Initializes the non-specific mode.

        Args:
            alphabet: The alphabet used to encode sequences into tensors.
            ignore_length: Whether to use the same non-specific binding factor
                regardless of input length.
            name: A string used to describe the non-specific mode.
        """
        super().__init__(
            out_channels=1, in_channels=len(alphabet.alphabet), name=name
        )
        self._layers: set[Conv0d]  # type: ignore[assignment]
        self.alphabet = alphabet
        self.ignore_length = ignore_length

    @override
    def __repr__(self) -> str:
        out = f"alphabet={repr(self.alphabet)}"
        if self.ignore_length:
            out += f", ignore_length={self.ignore_length})"
        return f"{type(self).__name__}({out})"

    @override
    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        del mode
        return length * 0 + 1

    @overload
    def in_len(self, length: T, mode: Literal["min"]) -> T: ...

    @overload
    def in_len(self, length: T, mode: Literal["max"]) -> None: ...

    @override
    def in_len(
        self, length: T, mode: Literal["min", "max"] = "max"
    ) -> T | None:
        if mode == "min":
            return length * 0 + 1
        return None


class Conv0d(Layer):
    r"""Non-specific '0D convolution' with length biases.

    Flattens the length dimension into a single value proportional to the
    number of finite elements in each sequence in log space.

    Attributes:
        log_posbias (Tensor): The bias for each input length
            :math:`\omega(|S_i|)`.
    """

    unfreezable = Literal[Layer.unfreezable, "posbias"]

    def __init__(
        self,
        nonspecific: NonSpecific,
        input_shape: int,
        min_input_length: int,
        max_input_length: int,
        train_posbias: bool = False,
    ) -> None:
        """Initializes the 0d convolution layer.

        Args:
            nonspecific: The specification of the 0d convolution layer.
            input_shape: The number of elements in an input sequence.
            min_input_length: The minimum number of finite elements in an input
                sequence.
            max_input_length: The maximum number of finite elements in an input
                sequence.
            train_posbias: Whether to train a bias for each input length.
        """
        super().__init__(
            layer_spec=nonspecific,
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
        )
        self.layer_spec: NonSpecific

        self.train_posbias = train_posbias
        n_lengths = self.max_input_length - self.min_input_length + 1
        self.log_posbias = torch.nn.Parameter(
            torch.zeros(size=(n_lengths, 1, 1), dtype=__precision__),
            requires_grad=train_posbias,
        )

    @classmethod
    def from_nonspecific(
        cls,
        nonspecific: NonSpecific,
        prev: Table[Any] | Layer,
        train_posbias: bool = False,
    ) -> Self:
        """Creates a new instance from a specification and an input component.

        Args:
            spec: The specification of the 0d convolution layer.
            prev: If used as the first layer, the table that will be passed as
                an input; otherwise, the layer that precedes it.
            train_posbias: Whether to train a bias for each input length.
        """
        if isinstance(prev, Layer):
            input_shape = prev.out_len(prev.input_shape, "shape")
            min_input_length = prev.out_len(prev.min_input_length, "min")
            max_input_length = prev.out_len(prev.max_input_length, "max")
        else:
            input_shape = prev.input_shape
            min_input_length = prev.min_read_length
            max_input_length = prev.max_read_length
        return cls(
            nonspecific=nonspecific,
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
            train_posbias=train_posbias,
        )

    @override
    def max_embedding_size(self) -> int:
        return (
            self.max_input_length
            * torch.tensor(data=[], dtype=torch.int64).element_size()
        )

    @override
    def check_length_consistency(self) -> None:
        super().check_length_consistency()
        alt_conv0d = Conv0d(
            nonspecific=NonSpecific(alphabet=self.layer_spec.alphabet),
            input_shape=self.input_shape,
            min_input_length=self.min_input_length,
            max_input_length=self.max_input_length,
        )
        if self.log_posbias.shape != alt_conv0d.log_posbias.shape:
            raise RuntimeError(
                f"expected posbias shape {alt_conv0d.log_posbias.shape}"
                f", found {self.log_posbias.shape}"
            )

    @override
    def unfreeze(self, parameter: unfreezable = "all") -> None:
        if parameter in ("posbias", "all") and self.train_posbias:
            self.log_posbias.requires_grad_()
        if parameter != "posbias":
            super().unfreeze(parameter)

    @override
    def update_binding_optim(
        self, binding_optim: BindingOptim
    ) -> BindingOptim:
        # Unfreeze posbias
        binding_optim = self.layer_spec.update_binding_optim(binding_optim)
        if self.train_posbias:
            binding_optim.steps.append(
                Step([Call(self, "unfreeze", {"parameter": "posbias"})])
            )
        return binding_optim

    @override
    def update_input_length(
        self,
        left_shift: int = 0,
        right_shift: int = 0,
        min_len_shift: int = 0,
        max_len_shift: int = 0,
        new_min_len: int | None = None,
        new_max_len: int | None = None,
    ) -> None:
        self.log_posbias = torch.nn.Parameter(
            F.pad(
                self.log_posbias, (0, 0, 0, 0, -min_len_shift, max_len_shift)
            ),
            requires_grad=self.log_posbias.requires_grad,
        )
        super().update_input_length(
            left_shift=left_shift,
            right_shift=right_shift,
            min_len_shift=min_len_shift,
            max_len_shift=max_len_shift,
            new_min_len=new_min_len,
            new_max_len=new_max_len,
        )

    def get_log_posbias(self) -> Tensor:
        r"""The bias for each input length :math:`\omega(|S_i|)`.

        Returns:
            A tensor with the bias of each input length of shape
            :math:`(\text{input_lengths},1,1)`.
        """
        return F.pad(self.log_posbias, (0, 0, 0, 0, self.min_input_length, 0))

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        r"""Calculates the log score of each sequence.

        .. math::
            \log \omega(|S_i|) + |S_i|

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            The log score tensor of shape :math:`(\text{minibatch},1,1)`.
        """
        lengths = self.lengths(seqs)
        if self.layer_spec.ignore_length:
            out = torch.zeros_like(lengths, dtype=__precision__)
        else:
            out = torch.log(lengths).to(__precision__)
        out = out.unsqueeze(1).unsqueeze(1)
        if self.log_posbias.requires_grad or torch.any(self.log_posbias != 0):
            out += self.get_log_posbias()[lengths]
        return out
