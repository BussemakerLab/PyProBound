"""ProBound convolutional models"""
from typing import Literal, TypeVar, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Any, Self, override

from . import __precision__
from .alphabet import Alphabet
from .base import BindingOptim, Call, Step
from .containers import Buffer
from .layers import Layer
from .psam import PSAM, NonSpecific
from .table import Table
from .utils import ceil_div

T = TypeVar("T", int, Tensor)


class Conv1d(Layer):
    """PSAM modeling with biases"""

    layer_spec: PSAM
    _unfreezable = Literal[Layer._unfreezable, "theta", "omega"]

    def __init__(
        self,
        psam: PSAM,
        input_shape: int,
        min_input_length: int,
        max_input_length: int,
        train_theta: bool = False,
        train_omega: bool = False,
        bias_bin: int = 1,
        one_hot: bool = False,
        normalize: bool = False,
    ) -> None:
        super().__init__(
            layer_spec=psam,
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
        )

        # check input
        if len(psam.symmetry) == 0 and (
            bias_bin > 1 or train_theta or one_hot
        ):
            raise ValueError(
                "If kernel_size is 0 interaction_distance must be"
                " 0, out_channels=bias_bin=1, train_theta=one_hot=False"
            )
        if min_input_length < 0:
            raise ValueError("min_input_length must be nonnegative")
        if min_input_length < len(psam.symmetry):
            raise ValueError("min_input_length must be at least kernel_size")

        # store instance attributes
        self._bias_bin: Tensor = Buffer(
            torch.tensor(bias_bin, dtype=torch.int32)
        )
        self._one_hot: Tensor = Buffer(torch.tensor(one_hot))
        self._normalize: Tensor = Buffer(torch.tensor(normalize))

        # create omega and theta parameters
        n_windows = self._num_windows(self.input_shape)
        n_lengths = self.max_input_length - self.min_input_length + 1
        self.train_theta = train_theta
        self.theta = torch.nn.Parameter(
            torch.zeros(
                size=(
                    n_lengths,
                    self.out_channels,
                    n_windows,
                    len(psam.symmetry),
                ),
                dtype=__precision__,
            ),
            requires_grad=train_theta,
        )
        self.train_omega = train_omega
        self.omega = torch.nn.Parameter(
            torch.zeros(
                size=(n_lengths, self.out_channels, n_windows),
                dtype=__precision__,
            ),
            requires_grad=train_omega,
        )

    @property
    def bias_bin(self) -> int:
        return cast(int, self._bias_bin.item())

    @property
    def one_hot(self) -> bool:
        return cast(bool, self._one_hot.item())

    @property
    def normalize(self) -> bool:
        return cast(bool, self._normalize.item())

    @override
    @property
    def out_channels(self) -> int:
        assert super().out_channels == self.layer_spec.out_channels
        return super().out_channels

    @override
    @property
    def in_channels(self) -> int:
        assert super().in_channels == self.layer_spec.in_channels
        return self.layer_spec.in_channels

    @classmethod
    def from_psam(
        cls,
        psam: PSAM,
        prev: Table[Any] | Layer,
        train_theta: bool = False,
        train_omega: bool = False,
        bias_bin: int = 1,
        one_hot: bool = False,
        normalize: bool = False,
    ) -> Self:
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
            train_theta=train_theta,
            train_omega=train_omega,
            bias_bin=bias_bin,
            one_hot=one_hot,
            normalize=normalize,
        )

    @override
    def check_length_consistency(self) -> None:
        super().check_length_consistency()
        alt_conv1d = Conv1d(
            psam=PSAM(
                kernel_size=len(self.layer_spec.symmetry),
                alphabet=self.layer_spec.alphabet,
                out_channels=self.layer_spec.out_channels,
                in_channels=self.layer_spec.in_channels,
                score_reverse=self.layer_spec.score_reverse,
            ),
            input_shape=self.input_shape,
            min_input_length=self.min_input_length,
            max_input_length=self.max_input_length,
            bias_bin=self.bias_bin,
        )
        if self.omega.shape != alt_conv1d.omega.shape:
            raise RuntimeError(
                f"expected omega shape {alt_conv1d.omega.shape}"
                f", found {self.omega.shape}"
            )
        if self.theta.shape != alt_conv1d.theta.shape:
            raise RuntimeError(
                f"expected theta shape {alt_conv1d.theta.shape}"
                f", found {self.theta.shape}"
            )

    @override
    def unfreeze(self, parameter: _unfreezable = "all") -> None:
        """Unfreeze the desired parameter"""
        if parameter in ("omega", "all") and self.train_omega:
            self.omega.requires_grad_()
        if parameter in ("theta", "all") and self.train_theta:
            self.theta.requires_grad_()
        # pylint: disable-next=consider-using-in
        if parameter != "omega" and parameter != "theta":
            super().unfreeze(parameter)

    @override
    def update_binding_optim(
        self, binding_optim: BindingOptim
    ) -> BindingOptim:
        # unfreeze omega/theta after mono / greedy search
        binding_optim = self.layer_spec.update_binding_optim(binding_optim)

        insertion_idx = 0
        for step_idx, step in enumerate(binding_optim.steps):
            for call in step.calls:
                if (
                    call.cmpt is self.layer_spec
                    and call.fun == "unfreeze"
                    and call.kwargs["parameter"] == "mono"
                ):
                    insertion_idx = max(step_idx + 1, insertion_idx)
                if call.fun in (
                    "update_footprint",
                    "update_read_length",
                    "shift_heuristic",
                ):
                    insertion_idx = max(step_idx + 1, insertion_idx)

        if self.train_omega:
            binding_optim.steps.insert(
                insertion_idx,
                Step([Call(self, "unfreeze", {"parameter": "omega"})]),
            )
            insertion_idx += 1
        if self.train_theta:
            binding_optim.steps.insert(
                insertion_idx,
                Step([Call(self, "unfreeze", {"parameter": "theta"})]),
            )

        binding_optim.merge_binding_optim()
        return binding_optim

    @override
    def max_embedding_size(self) -> int:
        """Size needed for embedding a sequence"""
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
        if self.layer_spec.interaction_distance > 0:
            total_in_channels **= 2
        return element_size * self.max_input_length * total_in_channels

    def _num_windows(self, input_length: int) -> int:
        """Number of sliding windows modeled by biases"""
        return ceil_div(self.out_len(input_length), self.bias_bin)

    def _update_biases(
        self,
        binding_mode_left: int = 0,
        binding_mode_right: int = 0,
        window_top: int = 0,
        window_bottom: int = 0,
        length_front: int = 0,
        length_back: int = 0,
    ) -> None:
        """Update theta and omega"""
        old_window_top, old_window_bottom = window_top, window_bottom
        window_top = self._num_windows(
            self.input_shape + old_window_top
        ) - self._num_windows(self.input_shape)
        window_bottom = self._num_windows(
            self.input_shape + old_window_top + old_window_bottom
        ) - self._num_windows(self.input_shape + old_window_top)
        self.omega = torch.nn.Parameter(
            F.pad(
                self.omega,
                (window_top, window_bottom, 0, 0, length_front, length_back),
            ),
            requires_grad=self.omega.requires_grad,
        )  # window dimensions is third instead of second
        self.theta = torch.nn.Parameter(
            F.pad(
                self.theta,
                (
                    binding_mode_left,
                    binding_mode_right,
                    window_top,
                    window_bottom,
                    0,
                    0,
                    length_front,
                    length_back,
                ),
            ),
            requires_grad=self.theta.requires_grad,
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

    def get_theta(self) -> Tensor:
        theta: Tensor = self.theta
        if self.normalize:
            theta = theta - theta.mean(dim=(-1, -2), keepdim=True)
        out = F.pad(
            theta.repeat_interleave(self.bias_bin, 2)[
                ..., : self.out_len(self.input_shape), :
            ],
            (0, 0, 0, 0, 0, 0, self.min_input_length, 0),
        )
        return 2 * torch.sigmoid(out)

    def get_omega(self) -> Tensor:
        omega: Tensor = self.omega
        if self.normalize:
            omega = omega - omega.mean(dim=-1, keepdim=True)
        out = F.pad(
            omega.repeat_interleave(self.bias_bin, -1)[
                ..., : self.out_len(self.input_shape)
            ],
            (0, 0, 0, 0, self.min_input_length, 0),
        )
        return out

    def score_onehot(
        self,
        seqs: Tensor,
        theta: Tensor | None = None,
        omega: Tensor | None = None,
    ) -> Tensor:
        """Score batch-first one-hot tensor of sequences using convolutions"""

        if len(seqs.shape) != 3:
            if self.layer_spec.alphabet is None:
                raise ValueError(
                    "sequences not embedded but alphabet is not specified"
                )
            current = self.layer_spec.alphabet.embedding.to(seqs.device)(
                seqs
            ).transpose(1, 2)
        else:
            current = seqs

        matrix = self.layer_spec.get_filter(0)

        if theta is None:
            if current.device.type == "cpu":
                # https://github.com/pytorch/pytorch/issues/104284
                result = F.conv3d(
                    current.unsqueeze(-1).unsqueeze(-1),
                    matrix.unsqueeze(-1).unsqueeze(-1),
                ).squeeze(-1, -2)
            else:
                result = F.conv1d(current, matrix)
        else:
            unfold = current.unfold(2, matrix.shape[2], 1)
            result = (unfold.unsqueeze(1) * matrix.unsqueeze(2)).sum(2)
            if theta is not None:
                result *= theta
            result = result.sum(3)

        for dist in range(1, self.layer_spec.interaction_distance + 1):
            matrix = self.layer_spec.get_filter(dist)
            if (not matrix.requires_grad) and (not torch.any(matrix != 0)):
                continue
            matrix = matrix.flatten(1, 2)

            if len(seqs.shape) != 3:
                if self.layer_spec.alphabet is None:
                    raise ValueError(
                        "sequences not embedded but alphabet is not specified"
                    )
                current_interactions = (
                    self.layer_spec.alphabet.interaction_embedding.to(
                        seqs.device
                    )(
                        seqs[:, :-dist]
                        * len(self.layer_spec.alphabet.embedding.weight)
                        + seqs[:, dist:]
                    ).transpose(
                        1, 2
                    )
                )
            else:
                current_interactions = (
                    current[:, :, dist:].unsqueeze(1)
                    * current[:, :, :-dist].unsqueeze(2)
                ).flatten(1, 2)

            if theta is None:
                if current.device.type == "cpu":
                    # https://github.com/pytorch/pytorch/issues/104284
                    result += F.conv3d(
                        current_interactions.unsqueeze(-1).unsqueeze(-1),
                        matrix.unsqueeze(-1).unsqueeze(-1),
                    ).squeeze(-1, -2)
                else:
                    result += F.conv1d(current_interactions, matrix)
            else:
                unfold = current_interactions.unfold(2, matrix.shape[2], 1)
                temp = (unfold.unsqueeze(1) * matrix.unsqueeze(2)).sum(2)
                if theta is not None:
                    temp *= torch.sqrt(theta[..., :-dist] * theta[..., dist:])
                result += temp.sum(3)

        if omega is not None:
            result += omega
        return (
            result.nan_to_num(
                nan=float("-inf"), neginf=float("-inf"), posinf=float("-inf")
            )
            + self.layer_spec.get_bias()
        )

    def score_dense(
        self,
        seqs: Tensor,
        theta: Tensor | None = None,
        omega: Tensor | None = None,
    ) -> Tensor:
        """Score a batch-first dense tensor of sequences using indexing"""

        # get matrix
        matrix = self.layer_spec.get_filter(0)
        matrix = F.pad(matrix, (0, 0, 0, 1), value=float("-inf"))

        # get sliding windows
        unfold = seqs.unfold(1, matrix.shape[-1], 1)

        # reshape and gather
        unfold_expand = unfold.unsqueeze(1).expand(-1, matrix.shape[0], -1, -1)
        matrix = matrix.expand(len(seqs), -1, -1, -1)
        gather = matrix.gather(dim=2, index=unfold_expand)

        # multiply in theta
        if theta is not None:
            gather *= theta

        # sum over features
        out = gather.sum(-1)

        # score interaction features
        for dist in range(1, self.layer_spec.interaction_distance + 1):
            # get matrix
            matrix = self.layer_spec.get_filter(dist)
            if (not matrix.requires_grad) and (not torch.any(matrix != 0)):
                continue
            matrix = F.pad(matrix, (0, 0, 0, 1, 0, 1), value=float("-inf"))

            # get sliding windows
            unfold2 = unfold.unfold(-1, dist + 1, 1)

            # flatten
            unfold2 = (unfold2[..., 0] * matrix.shape[-2]) + unfold2[..., -1]
            matrix = matrix.flatten(-3, -2)

            # reshape and gather
            unfold2 = unfold2.unsqueeze(1).expand(-1, matrix.shape[-3], -1, -1)
            matrix = matrix.expand(len(seqs), -1, -1, -1)
            if matrix.device.type == "mps" and matrix.shape[-1] == 1:
                # https://github.com/pytorch/pytorch/issues/94765
                gather = matrix.expand(-1, -1, -1, 2).gather(
                    dim=2, index=unfold2.expand(-1, -1, -1, 2)
                )[..., :1]
            else:
                gather = matrix.gather(dim=2, index=unfold2)

            # multiply in theta
            if theta is not None:
                gather = gather * torch.sqrt(
                    theta[..., :-dist] * theta[..., dist:]
                )

            # sum over features
            out += gather.sum(-1)

        # add in omega
        if omega is not None:
            out += omega

        return out + self.layer_spec.get_bias()

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        """Score a batch-first tensor of sequences"""

        # check input
        requires_embedding = True
        if seqs.ndim == 2:
            requires_embedding = cast(
                bool,
                torch.any(
                    seqs
                    > len(cast(Alphabet, self.layer_spec.alphabet).alphabet)
                ).item(),
            )
        if len(seqs) > 1 and not self.one_hot:  # used for calculating E[score]
            if seqs.ndim == 3:
                raise ValueError(
                    "Input has 3 dimensions, but one_hot not enabled"
                )
            if requires_embedding:
                raise ValueError(
                    "Input requires embedding, but one_hot not enabled"
                )

        # get theta and omega
        lengths, theta, omega = None, None, None
        if self.theta.requires_grad or torch.any(self.theta != 0):
            if lengths is None:
                lengths = self.lengths(seqs)
            theta = self.get_theta()[lengths]
        if self.omega.requires_grad or torch.any(self.omega != 0):
            if lengths is None:
                lengths = self.lengths(seqs)
            omega = self.get_omega()[lengths]

        # score
        if self.one_hot or seqs.ndim == 3 or requires_embedding:
            return self.score_onehot(seqs, theta, omega)
        return self.score_dense(seqs, theta, omega)


class Conv0d(Layer):
    """Non-specific '0D convolution' with biases"""

    layer_spec: NonSpecific
    _unfreezable = Literal[Layer._unfreezable, "omega"]

    def __init__(
        self,
        nonspecific: NonSpecific,
        input_shape: int,
        min_input_length: int,
        max_input_length: int,
        train_omega: bool = False,
    ) -> None:
        super().__init__(
            layer_spec=nonspecific,
            input_shape=input_shape,
            min_input_length=min_input_length,
            max_input_length=max_input_length,
        )

        self.train_omega = train_omega
        n_lengths = self.max_input_length - self.min_input_length + 1
        self.omega = torch.nn.Parameter(
            torch.zeros(
                size=(n_lengths, self.out_channels, 1), dtype=__precision__
            ),
            requires_grad=train_omega,
        )

    @override
    @property
    def out_channels(self) -> int:
        assert super().out_channels == self.layer_spec.out_channels
        return super().out_channels

    @override
    @property
    def in_channels(self) -> int:
        assert super().in_channels == self.layer_spec.in_channels
        return self.layer_spec.in_channels

    @classmethod
    def from_nonspecific(
        cls,
        nonspecific: NonSpecific,
        prev: Table[Any] | Layer,
        train_omega: bool = False,
    ) -> Self:
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
            train_omega=train_omega,
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
        if self.omega.shape != alt_conv0d.omega.shape:
            raise RuntimeError(
                f"expected omega shape {alt_conv0d.omega.shape}"
                f", found {self.omega.shape}"
            )

    @override
    def unfreeze(self, parameter: _unfreezable = "all") -> None:
        """Unfreeze the desired parameter"""
        if parameter in ("omega", "all") and self.train_omega:
            self.omega.requires_grad_()
        if parameter != "omega":
            super().unfreeze(parameter)

    @override
    def update_binding_optim(
        self, binding_optim: BindingOptim
    ) -> BindingOptim:
        # unfreeze omega
        binding_optim = self.layer_spec.update_binding_optim(binding_optim)
        if self.train_omega:
            binding_optim.steps.append(
                Step([Call(self, "unfreeze", {"parameter": "omega"})])
            )
        return binding_optim

    @override
    def max_embedding_size(self) -> int:
        """Size needed for embedding a sequence"""
        element_size = torch.tensor(data=[], dtype=torch.int64).element_size()
        total_in_channels = 1
        return element_size * self.max_input_length * total_in_channels

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
        self.omega = torch.nn.Parameter(
            F.pad(self.omega, (0, 0, 0, 0, -min_len_shift, max_len_shift)),
            requires_grad=self.omega.requires_grad,
        )
        super().update_input_length(
            left_shift=left_shift,
            right_shift=right_shift,
            min_len_shift=min_len_shift,
            max_len_shift=max_len_shift,
            new_min_len=new_min_len,
            new_max_len=new_max_len,
        )

    def get_omega(self) -> Tensor:
        return F.pad(self.omega, (0, 0, 0, 0, self.min_input_length, 0))

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        lengths = self.lengths(seqs)
        out = torch.log(lengths).unsqueeze(1).unsqueeze(1)
        if self.omega.requires_grad or torch.any(self.omega != 0):
            out += self.get_omega()[lengths]
        return out
