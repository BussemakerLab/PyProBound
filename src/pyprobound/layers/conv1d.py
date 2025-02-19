"""1d convolution implementation for a PSAM model.

Members are explicitly re-exported in pyprobound.layers.
"""

from collections.abc import Sequence
from typing import Any, Literal, TypeVar, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.module import _addindent
from typing_extensions import Self, override

from .. import __precision__
from ..base import BindingOptim, Call, Step
from ..table import Table
from ..utils import ceil_div
from .layer import Layer
from .psam import PSAM

T = TypeVar("T", int, Tensor)


class Conv1d(Layer):
    r"""1d convolution with a PSAM filter and position bias modeling.

    Since the weight :math:`\beta_\phi` of feature :math:`\phi` is defined as
    :math:`-\Delta\Delta G_\phi/RT`, the output of the convolution is the
    :math:`-\log K^{rel}_{\text{D}}` of each sliding window.

    .. math::
            \log \frac{1}{K^{rel}_{\text{D}, a} (S_{i, x})}
            = \omega(x) + \sum_{\phi} \beta_\phi \mathbb{1}_\phi(S_{i, x})
    where :math:`\mathbb{1}_\phi(S_{i, x})` is the indicator function of when
    window :math:`x` of sequence :math:`i` contains feature :math:`\phi`.

    Attributes:
        log_posbias (Tensor): The bias :math:`\omega(x)` for each output
            position and channel.
    """

    unfreezable = Literal[Layer.unfreezable, "posbias"]

    def __init__(
        self,
        psam: PSAM,
        input_shape: int,
        min_input_length: int,
        max_input_length: int,
        train_posbias: bool = False,
        bias_mode: Literal["channel", "same", "reverse"] = "channel",
        bias_bin: int = 1,
        length_specific_bias: bool = True,
        out_channel_indexing: Sequence[int] | None = None,
        one_hot: bool = False,
        unfold: bool = False,
        normalize: bool = False,
    ) -> None:
        r"""Initializes the 1d convolution layer.

        Args:
            psam: The specification of the 1d convolution layer.
            input_shape: The number of elements in an input sequence.
            min_input_length: The minimum number of finite elements in an input
                sequence.
            max_input_length: The maximum number of finite elements in an input
                sequence.
            train_posbias: Whether to train a bias :math:`\omega(x)` for each
                output position and channel.
            bias_mode: Whether to train a separate bias for each output
                channel, use the same bias across all output channels, or (if
                `score_reverse`) flip it for the reverse output channels.
            bias_bin: Applies the constraint
                :math:`\omega(x_{i\times\text{bias_bin}}) = \cdots
                = \omega(x_{(i+1)\times\text{bias_bin}-1})`.
            length_specific_bias: Whether to train a separate bias parameter
                for each input length.
            out_channel_indexing: Output channel indexing, equivalent to
                `Conv1d(seqs)[:,out_channel_indexing]`.
            one_hot: Whether to use one-hot scoring instead of dense.
            unfold: Whether to score using `unfold` or `conv1d` (if `one_hot`).
            normalize: Whether to mean-center `log_posbias` over all windows.
        """
        super().__init__(
            layer_spec=psam,
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
        )
        self.layer_spec: PSAM

        # Check input
        if min_input_length < 0:
            raise ValueError("min_input_length must be nonnegative")
        if min_input_length < self.layer_spec.kernel_size:
            raise ValueError("min_input_length must be at least kernel_size")
        if bias_mode == "reverse" and not self.layer_spec.score_reverse:
            raise ValueError(
                "Cannot use bias_mode reverse if not score_reverse"
            )

        # Store instance attributes
        self.one_hot = one_hot
        self.unfold = unfold
        self.normalize = normalize
        self._bias_mode = bias_mode
        self._bias_bin = bias_bin
        self._length_specific_bias = length_specific_bias
        self._out_channel_indexing = (
            list(out_channel_indexing)
            if out_channel_indexing is not None
            else None
        )

        # Create posbias parameter
        n_windows = self._num_windows(self.input_shape)
        if self.length_specific_bias:
            n_lengths = self.max_input_length - self.min_input_length + 1
        else:
            n_lengths = 1
        if self.bias_mode == "channel":
            bias_channels = self.layer_spec.out_channels
        elif self.bias_mode == "same":
            bias_channels = 1
        elif self.bias_mode == "reverse":
            bias_channels = self.layer_spec.out_channels // 2
        self.train_posbias = train_posbias
        self.log_posbias = torch.nn.Parameter(
            torch.zeros(
                size=(
                    n_lengths,
                    bias_channels,  # pylint: disable=possibly-used-before-assignment
                    n_windows,
                ),
                dtype=__precision__,
            ),
            requires_grad=train_posbias,
        )

    @override
    def __repr__(self) -> str:
        args = [repr(self.layer_spec)]
        if (
            self.min_input_length != self.max_input_length
            and not self.length_specific_bias
        ):
            args.append(f"length_specific_bias={self.length_specific_bias}")
        if self.bias_mode != "channel":
            args.append(f"bias_mode={self.bias_mode}")
        if self.bias_bin != 1:
            args.append(f"bias_bin={self.bias_bin}")
        if self.out_channel_indexing is not None:
            args.append(f"out_channel_indexing={self.out_channel_indexing}")
        if self.one_hot:
            args.append(f"one_hot={self.one_hot}")
        if self.unfold:
            args.append(f"unfold={self.unfold}")

        out = []
        temp = ""
        for i in args:
            if len(temp) == 0:
                temp = i
            elif len(temp) + len(i) + 2 > 60:
                out.append(temp + ",")
                temp = i
            else:
                temp += ", " + i
        out.append(temp)
        if len(out) > 1 or "\n" in out[0]:
            return (
                f"{type(self).__name__}(\n  "
                + "\n  ".join(_addindent(i, 2) for i in out)  # type: ignore[no-untyped-call]
                + "\n)"
            )
        return f"{type(self).__name__}( {out[0]} )"

    @classmethod
    def from_psam(
        cls,
        psam: PSAM,
        prev: Table[Any] | Layer,
        train_posbias: bool = False,
        bias_mode: Literal["channel", "same", "reverse"] = "channel",
        bias_bin: int = 1,
        length_specific_bias: bool = True,
        out_channel_indexing: Sequence[int] | None = None,
        one_hot: bool = False,
        unfold: bool = False,
        normalize: bool = False,
    ) -> Self:
        r"""Creates a new instance from a PSAM and an input component.

        Args:
            psam: The specification of the 1d convolution layer.
            prev: If used as the first layer, the table that will be passed as
                an input; otherwise, the layer that precedes it.
            train_posbias: Whether to train a bias :math:`\omega(x)` for each
                output position and channel.
            bias_mode: Whether to train a separate bias for each output
                channel, use the same bias across all output channels, or (if
                `score_reverse`) flip it for the reverse output channels.
            bias_bin: Applies the constraint
                :math:`\omega(x_{i\times\text{bias_bin}}) = \cdots
                = \omega(x_{(i+1)\times\text{bias_bin}-1})`.
            length_specific_bias: Whether to train a separate bias parameter
                for each input length.
            out_channel_indexing: Output channel indexing, equivalent to
                `Conv1d(seqs)[:,out_channel_indexing]`.
            one_hot: Whether to use one-hot scoring instead of dense.
            unfold: Whether to score using `unfold` or `conv1d` (if `one_hot`).
            normalize: Whether to mean-center `log_posbias` over all windows.
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
            psam=psam,
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
            train_posbias=train_posbias,
            bias_mode=bias_mode,
            bias_bin=bias_bin,
            length_specific_bias=length_specific_bias,
            out_channel_indexing=out_channel_indexing,
            one_hot=one_hot,
            unfold=unfold,
            normalize=normalize,
        )

    @property
    def bias_mode(self) -> Literal["channel", "same", "reverse"]:
        """Whether to train a separate bias for each output
        channel, use the same bias across all output channels, or (if
        `score_reverse`) flip it for the reverse output channels."""
        return self._bias_mode

    @property
    def bias_bin(self) -> int:
        r"""Applies the constraint
        :math:`\omega(x_{i\times\text{bias_bin}})
        = \cdots = \omega(x_{(i+1)\times\text{bias_bin}-1})`.
        """
        return self._bias_bin

    @property
    def length_specific_bias(self) -> bool:
        """Whether to train a separate bias parameter for each input length."""
        return self._length_specific_bias

    @property
    def out_channel_indexing(self) -> list[int] | None:
        """Output channel indexing, equivalent to
        `Conv1d(seqs)[:,out_channel_indexing]`."""
        return self._out_channel_indexing

    @override
    @property
    def out_channels(self) -> int:
        if self.out_channel_indexing is not None:
            return len(self.out_channel_indexing)
        return super().out_channels

    def _num_windows(self, input_length: int) -> int:
        """The number of sliding windows modeled by biases."""
        return ceil_div(self.out_len(input_length), self.bias_bin)

    @override
    def check_length_consistency(self) -> None:
        super().check_length_consistency()
        alt_conv1d = Conv1d(
            psam=PSAM(
                kernel_size=self.layer_spec.kernel_size,
                alphabet=self.layer_spec.alphabet,
                out_channels=self.layer_spec.out_channels,
                in_channels=self.layer_spec.in_channels,
                score_reverse=self.layer_spec.score_reverse,
                dilation=self.layer_spec.dilation,
            ),
            input_shape=self.input_shape,
            min_input_length=self.min_input_length,
            max_input_length=self.max_input_length,
            bias_mode=self.bias_mode,
            bias_bin=self.bias_bin,
            length_specific_bias=self.length_specific_bias,
            out_channel_indexing=self.out_channel_indexing,
        )
        if self.log_posbias.shape != alt_conv1d.log_posbias.shape:
            raise RuntimeError(
                f"expected posbias shape {alt_conv1d.log_posbias.shape}"
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
        # Unfreeze posbias after monomer / greedy search
        binding_optim = self.layer_spec.update_binding_optim(binding_optim)

        insertion_idx = 0
        for step_idx, step in enumerate(binding_optim.steps):
            for call in step.calls:
                if (
                    call.cmpt is self.layer_spec
                    and call.fun == "unfreeze"
                    and call.kwargs["parameter"] == "monomer"
                ):
                    insertion_idx = max(step_idx + 1, insertion_idx)
                if call.fun in (
                    "update_footprint",
                    "update_read_length",
                    "shift_heuristic",
                ):
                    insertion_idx = max(step_idx + 1, insertion_idx)

        if self.train_posbias:
            binding_optim.steps.insert(
                insertion_idx,
                Step([Call(self, "unfreeze", {"parameter": "posbias"})]),
            )
            insertion_idx += 1

        binding_optim.merge_binding_optim()
        return binding_optim

    @override
    def max_embedding_size(self) -> int:
        if self.one_hot:
            element_size = torch.tensor(
                data=[], dtype=__precision__
            ).element_size()
            total_in_channels = self.in_channels
        else:
            element_size = torch.tensor(
                data=[], dtype=torch.int64
            ).element_size()
            total_in_channels = 1
        if self.layer_spec.pairwise_distance > 0:
            total_in_channels **= 2
        return element_size * self.max_input_length * total_in_channels

    def _update_biases(
        self,
        binding_mode_left: int = 0,
        binding_mode_right: int = 0,
        window_top: int = 0,
        window_bottom: int = 0,
        length_front: int = 0,
        length_back: int = 0,
    ) -> None:
        """Updates biases according to specified padding values."""
        del binding_mode_left, binding_mode_right
        if not self.length_specific_bias:
            length_front, length_back = 0, 0
        old_window_top, old_window_bottom = window_top, window_bottom
        window_top = self._num_windows(
            self.input_shape + old_window_top
        ) - self._num_windows(self.input_shape)
        window_bottom = self._num_windows(
            self.input_shape + old_window_top + old_window_bottom
        ) - self._num_windows(self.input_shape + old_window_top)
        self.log_posbias = torch.nn.Parameter(
            F.pad(
                self.log_posbias,
                (window_top, window_bottom, 0, 0, length_front, length_back),
            ),
            requires_grad=self.log_posbias.requires_grad,
        )

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
        """Updates input shapes, called by a child PSAM after its update.

        Args:
            left_shift: The change in size on the left side of the sequence.
            right_shift: The change in size on the right side of the sequence.
            min_len_shift: The change in the number of short input lengths.
            max_len_shift: The change in the number of long input lengths.
            new_min_len: The new `min_input_length`.
            new_max_len: The new `max_input_length`.
        """
        self._update_biases(
            window_top=left_shift,
            window_bottom=right_shift,
            length_front=-min_len_shift,
            length_back=max_len_shift,
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
        r"""The bias :math:`\omega(x)` for each output position and channel.

        Returns:
            A tensor with the bias of each output position and channel of shape
            :math:`(\text{input_lengths},\text{out_channels},
            \text{out_length})`.
        """
        log_posbias: Tensor = self.log_posbias
        if self.normalize:
            log_posbias = log_posbias - log_posbias.mean(dim=-1, keepdim=True)
        if self.bias_bin > 1:
            log_posbias = log_posbias.repeat_interleave(self.bias_bin, -1)[
                ..., : self.out_len(self.input_shape)
            ]
        if self.bias_mode == "reverse":
            log_posbias = torch.cat(
                (log_posbias, log_posbias.flip(-1, -2)), dim=1
            )
        if self.length_specific_bias:
            log_posbias = F.pad(
                log_posbias, (0, 0, 0, 0, self.min_input_length, 0)
            )
        return log_posbias

    def _get_log_posbias_indexed(self, seqs: Tensor) -> Tensor | None:
        r"""The bias :math:`\omega(x)` for each sequence.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{in_length})`.

        Returns:
            A tensor with the bias for each sequence of shape
            :math:`(\text{input_lengths},\text{out_channels},
            \text{out_length})`, or None if not being trained.
        """
        if self.log_posbias.requires_grad or torch.any(self.log_posbias != 0):
            posbias = self.get_log_posbias()
            if self.length_specific_bias:
                return posbias[self.lengths(seqs)]
            return posbias
        return None

    @staticmethod
    def _masked_conv1d(
        seq: Tensor, weight: Tensor, dilation: int = 1
    ) -> Tensor:
        """Convolve non-finite inputs without breaking gradient calculation."""

        # https://github.com/pytorch/pytorch/issues/104284
        # https://github.com/pytorch/pytorch/issues/96225
        def _conv1d(seq: Tensor, weight: Tensor, dilation: int) -> Tensor:
            return F.conv3d(
                seq.unsqueeze(-1).unsqueeze(-1),
                weight.unsqueeze(-1).unsqueeze(-1),
                dilation=dilation,
            ).squeeze(-1, -2)

        return torch.where(
            torch.isfinite(_conv1d(seq, weight, dilation)),
            _conv1d(seq.masked_fill(~seq.isfinite(), 0), weight, dilation),
            float("-inf"),
        )

    @staticmethod
    def _masked_mul(seq: Tensor, weight: Tensor) -> Tensor:
        """Multiply non-finite inputs without breaking gradient calculation."""
        return (seq.masked_fill(~seq.isfinite(), 0) * weight).masked_fill(
            ~seq.isfinite(), float("-inf")
        )

    def score_onehot(self, seqs: Tensor) -> Tensor:
        r"""Calculates the log score of each window using convolutions.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{in_length})`.

        Returns:
            A tensor with the log score of each window of shape
            :math:`(\text{minibatch},\text{out_channels},\text{out_length})`.
        """

        if seqs.ndim != 3:
            if self.layer_spec.alphabet is None:
                raise ValueError(
                    "sequences not embedded but alphabet is not specified"
                )
            current = self.layer_spec.alphabet.embed(seqs)
        else:
            current = seqs

        matrix = self.layer_spec.get_filter(0)
        if self.out_channel_indexing is not None:
            matrix = matrix[self.out_channel_indexing]

        if self.unfold:
            unfold = current.unfold(
                2, self.layer_spec.dilation * (matrix.shape[2] - 1) + 1, 1
            )[..., :: self.layer_spec.dilation]
            result = self._masked_mul(
                unfold.unsqueeze(1), matrix.unsqueeze(2)
            ).sum(2)
            result = result.sum(3)
        else:
            result = self._masked_conv1d(
                current, matrix, dilation=self.layer_spec.dilation
            )

        for dist in range(1, self.layer_spec.pairwise_distance + 1):
            matrix = self.layer_spec.get_filter(dist)
            if self.out_channel_indexing is not None:
                matrix = matrix[self.out_channel_indexing]
            if (not matrix.requires_grad) and (not torch.any(matrix != 0)):
                continue
            matrix = matrix.flatten(1, 2)

            if seqs.ndim != 3:
                if self.layer_spec.alphabet is None:
                    raise ValueError(
                        "sequences not embedded but alphabet is not specified"
                    )
                current_pairs = self.layer_spec.alphabet.pairwise_embed(
                    seqs, dist
                )
            else:
                current_pairs = (
                    current[:, :, dist:].unsqueeze(1)
                    * current[:, :, :-dist].unsqueeze(2)
                ).flatten(1, 2)

            if self.unfold:
                unfold = current_pairs.unfold(2, matrix.shape[2], 1)
                temp = self._masked_mul(
                    unfold.unsqueeze(1), matrix.unsqueeze(2)
                ).sum(2)
                result += temp.sum(3)
            else:
                result += self._masked_conv1d(current_pairs, matrix)

        # Add in posbias
        posbias = self._get_log_posbias_indexed(seqs)
        if posbias is not None:
            result += posbias

        # Add in PSAM bias
        bias = self.layer_spec.get_bias()
        if self.out_channel_indexing is not None:
            bias = bias[self.out_channel_indexing]
        return result + bias

    def score_dense(self, seqs: Tensor) -> Tensor:
        r"""Calculates the log score of each window using indexing.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})`.

        Returns:
            A tensor with the log score of each window of shape
            :math:`(\text{minibatch},\text{out_channels},\text{out_length})`.
        """

        # Get matrix
        matrix = self.layer_spec.get_filter(0)
        if self.out_channel_indexing is not None:
            matrix = matrix[self.out_channel_indexing]
        matrix = F.pad(matrix, (0, 0, 0, 1), value=float("-inf"))
        matrix = torch.cat(
            [matrix, matrix[:, :-1].mean(1, keepdim=True)], dim=1
        )
        matrix = F.pad(matrix, (0, 0, 0, 1))

        # Get sliding windows
        unfold = seqs.unfold(
            1, self.layer_spec.dilation * (matrix.shape[-1] - 1) + 1, 1
        )[..., :: self.layer_spec.dilation]

        # Reshape and gather
        unfold_expand = unfold.unsqueeze(1).expand(-1, matrix.shape[0], -1, -1)
        matrix = matrix.expand(len(seqs), -1, -1, -1)
        gather = matrix.gather(dim=2, index=unfold_expand)

        # Sum over features
        out = gather.sum(-1)

        # Score pairwise features
        for dist in range(1, self.layer_spec.pairwise_distance + 1):
            # Get matrix
            matrix = self.layer_spec.get_filter(dist)
            if (not matrix.requires_grad) and (not torch.any(matrix != 0)):
                continue
            matrix = F.pad(matrix, (0, 0, 0, 1, 0, 1), value=float("-inf"))
            matrix = torch.cat(
                [matrix, matrix[:, :-1].mean(1, keepdim=True)], dim=1
            )
            matrix = torch.cat(
                [matrix, matrix[:, :, :-1].mean(2, keepdim=True)], dim=2
            )
            matrix = F.pad(matrix, (0, 0, 0, 1, 0, 1))
            if self.out_channel_indexing is not None:
                matrix = matrix[self.out_channel_indexing]

            # Get sliding windows
            unfold2 = unfold.unfold(-1, dist + 1, 1)

            # Flatten
            unfold2 = (unfold2[..., 0] * matrix.shape[-2]) + unfold2[..., -1]
            matrix = matrix.flatten(-3, -2)

            # Reshape and gather
            unfold2 = unfold2.unsqueeze(1).expand(-1, matrix.shape[-3], -1, -1)
            matrix = matrix.expand(len(seqs), -1, -1, -1)
            if matrix.device.type == "mps" and matrix.shape[-1] == 1:
                # https://github.com/pytorch/pytorch/issues/94765
                gather = matrix.expand(-1, -1, -1, 2).gather(
                    dim=2, index=unfold2.expand(-1, -1, -1, 2)
                )[..., :1]
            else:
                gather = matrix.gather(dim=2, index=unfold2)

            # Sum over features
            out += gather.sum(-1)

        # Add in posbias
        posbias = self._get_log_posbias_indexed(seqs)
        if posbias is not None:
            out += posbias

        # Add in PSAM bias
        bias = self.layer_spec.get_bias()
        if self.out_channel_indexing is not None:
            bias = bias[self.out_channel_indexing]
        return out + bias

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        r"""Calculates the log score of each window.

        .. math::
            \log \frac{1}{K^{rel}_{\text{D}, a} (S_{i, x})}
            = \omega(x) + \sum_{\phi} \beta_\phi \mathbb{1}_\phi(S_{i, x})

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{in_length})`.

        Returns:
            A tensor with the log score of each window of shape
            :math:`(\text{minibatch},\text{out_channels},\text{out_length})`.
        """

        # Check input
        requires_embedding = True
        if seqs.ndim == 2:
            if self.layer_spec.alphabet is None:
                raise ValueError(
                    "sequences not embedded but alphabet is not specified"
                )
            requires_embedding = cast(
                bool,
                torch.any(
                    seqs
                    > len(self.layer_spec.alphabet.alphabet) + 2
                    # Can be ' '=neginf, '*'=uninformative, '-'=zero
                ).item(),
            )
        if len(seqs) > 1 and not self.one_hot:  # Used for calculating E[score]
            if seqs.ndim == 3:
                raise ValueError(
                    "Input has 3 dimensions, but one_hot not enabled"
                )
            if requires_embedding:
                raise ValueError(
                    "Input requires embedding, but one_hot not enabled"
                )

        # Score
        if self.one_hot or seqs.ndim == 3 or requires_embedding:
            return self.score_onehot(seqs)
        return self.score_dense(seqs)
