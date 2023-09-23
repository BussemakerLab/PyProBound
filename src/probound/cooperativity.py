"""Binding mode interactions"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self, override

from . import __precision__
from .base import Binding, BindingOptim, Call, Component, Spec, Step, Transform
from .binding import BindingMode
from .containers import Buffer
from .layers import BindingModeKey
from .psam import PSAM
from .utils import ceil_div


class Spacing(Spec):
    """Dimer interaction parameters"""

    _unfreezable = Literal[Spec._unfreezable, "spacing"]

    def __init__(
        self,
        binding_mode_key_0: BindingModeKey,
        binding_mode_key_1: BindingModeKey,
        max_spacing: int | None = None,
        max_overlap: int | None = None,
        normalize: bool = False,
        name: str = "",
    ) -> None:
        super().__init__(name=name)

        if max_overlap is not None and max_overlap < 0:
            raise ValueError("max_overlap must be >= 0")
        if max_spacing is not None and max_spacing < 0:
            raise ValueError("max_overlap must be >= 0")

        self.binding_mode_key_0 = binding_mode_key_0
        self.binding_mode_key_1 = binding_mode_key_1
        self.flattened_spacing = torch.nn.Parameter(
            torch.zeros(size=(self.n_strands, 1), dtype=__precision__)
        )
        self._max_overlap: Tensor = Buffer(
            torch.tensor(float("inf") if max_overlap is None else max_overlap)
        )
        self._max_spacing: Tensor = Buffer(
            torch.tensor(float("inf") if max_spacing is None else max_spacing)
        )
        self._normalize: Tensor = Buffer(torch.tensor(normalize))
        self._binding_cooperativity: set[BindingCooperativity] = set()

    @property
    def normalize(self) -> bool:
        return cast(bool, self._normalize.item())

    @property
    def max_overlap(self) -> int | None:
        if torch.isinf(self._max_overlap):
            return None
        return cast(int, self._max_overlap.item())

    @property
    def max_spacing(self) -> int | None:
        if torch.isinf(self._max_spacing):
            return None
        return cast(int, self._max_spacing.item())

    @classmethod
    def from_psams(
        cls,
        psam_0: PSAM,
        psam_1: PSAM,
        max_spacing: int | None = None,
        max_overlap: int | None = None,
        name: str = "",
    ) -> Self:
        return cls(
            BindingModeKey([psam_0]),
            BindingModeKey([psam_1]),
            max_spacing=max_spacing,
            max_overlap=max_overlap,
            name=name,
        )

    @property
    def max_num_windows(self) -> int:
        return (self.flattened_spacing.shape[-1] + 1) // 2

    @property
    def n_strands(self) -> int:
        if (
            self.binding_mode_key_0[-1].out_channels not in (1, 2)
            or self.binding_mode_key_0[-1].out_channels
            != self.binding_mode_key_1[-1].out_channels
        ):
            raise RuntimeError(
                f"{self.binding_mode_key_0} has"
                f" {self.binding_mode_key_0[-1].out_channels} strands"
                f" but {self.binding_mode_key_1} has"
                f" {self.binding_mode_key_1[-1].out_channels} strands"
            )
        return self.binding_mode_key_0[-1].out_channels

    @override
    def components(self) -> Iterator[Component]:
        return iter(())

    @override
    def unfreeze(self, parameter: _unfreezable = "all") -> None:
        """Unfreeze the desired parameter"""
        if parameter in ("spacing", "all"):
            self.flattened_spacing.requires_grad_()
        if parameter != "spacing":
            super().unfreeze(parameter)

    @override
    def update_binding_optim(
        self, binding_optim: BindingOptim
    ) -> BindingOptim:
        # unfreeze spacing with first mono of Conv1d if available
        insertion_idx = len(binding_optim.steps)
        append = False
        for step_idx, step in enumerate(binding_optim.steps):
            for call in step.calls:
                if call.fun == "alpha_heuristic":
                    insertion_idx = step_idx + 1
                if call.fun == "unfreeze":
                    if call.kwargs["parameter"] in ("mono", "spacing"):
                        insertion_idx = min(step_idx, insertion_idx)
                        append = True

        new_call = Call(self, "unfreeze", {"parameter": "spacing"})
        if append:
            binding_optim.steps[insertion_idx].calls.append(new_call)
        else:
            binding_optim.steps.insert(insertion_idx, Step([new_call]))

        return binding_optim

    def _update_propagation(self) -> None:
        new_max_num_windows = max(
            max(coop.out_shape_0, coop.out_shape_1)
            for coop in self._binding_cooperativity
        )
        if new_max_num_windows < 1:
            raise RuntimeError("Cannot shrink spacing to less than 1 window")
        shift = new_max_num_windows - self.max_num_windows
        self.flattened_spacing = torch.nn.Parameter(
            F.pad(self.flattened_spacing, (shift, shift)),
            requires_grad=self.flattened_spacing.requires_grad,
        )

    def get_spacing(self, n_windows_0: int, n_windows_1: int) -> Tensor:
        """Diagonal-constant spacing matrix for each strand"""
        size_0 = self.binding_mode_key_0.in_len(1, mode="max")
        size_1 = self.binding_mode_key_1.in_len(1, mode="max")
        if size_0 is None or size_1 is None:
            raise RuntimeError(
                f"Receptive field of {self} is calculated as"
                f" {size_0} and {size_1}; expected int for both"
            )

        # normalize flattened_spacing
        spacing: Tensor = self.flattened_spacing
        if self.normalize:
            spacing = spacing - spacing.mean()

        # shift to the right index
        shift_left = self.max_num_windows - n_windows_0
        shift_right = n_windows_1 - self.max_num_windows
        if shift_right == 0:
            spacing = spacing[:, shift_left:]
        else:
            spacing = spacing[:, shift_left:shift_right]
        assert spacing.shape == (self.n_strands, n_windows_0 + n_windows_1 - 1)

        # convert to diagonal constant matrix
        spacing = spacing.unfold(-1, n_windows_1, 1).flip(-2)
        assert spacing.shape == (self.n_strands, n_windows_0, n_windows_1)

        # adjust for max_overlap and max_spacing
        # overlap(offset) = min(
        #   min(size_0, size_1),
        #   size_0 - offset if offset > 0 else size_1 + offset
        # )
        # spacing = -overlap(offset)
        if self.max_spacing is not None:
            for offset in range(
                self.max_spacing + size_0 + 1, spacing.shape[-1] + 1
            ):  # positive offsets
                spacing.diagonal(offset, -2, -1).fill_(float("-inf"))
            for offset in range(
                self.max_spacing + size_1 + 1, spacing.shape[-2] + 1
            ):  # negative offsets
                spacing.diagonal(-offset, -2, -1).fill_(float("-inf"))
        if self.max_overlap is not None and self.max_overlap < min(
            size_0, size_1
        ):
            for offset in range(
                -size_1 + self.max_overlap + 1, size_0 - self.max_overlap
            ):
                spacing.diagonal(offset, -2, -1).fill_(float("-inf"))

        # stack strands into a single matrix
        if self.n_strands == 2:
            # I  : bm_0_for, bm_1_rev
            # II : bm_0_for, bm_1_for
            # III: bm_0_rev, bm_1_for
            # IV : bm_0_rev, bm_1_rev
            spacing = torch.cat(
                (spacing.flatten(-3, -2), spacing.flip(-3).flatten(-3, -2)), -1
            )
        else:
            spacing = spacing.squeeze(-3)
        assert spacing.shape == (
            self.n_strands * n_windows_0,
            self.n_strands * n_windows_1,
        )

        return spacing


class BindingCooperativity(Binding):
    """Dimer interaction modeling with bias"""

    _unfreezable = Literal[Binding._unfreezable, "hill", "omega"]
    _cache_fun = "score_windows"

    def __init__(
        self,
        spacing: Spacing,
        binding_mode_0: BindingMode,
        binding_mode_1: BindingMode,
        train_omega: bool = False,
        train_hill: bool = False,
        bias_bin: int = 1,
        normalize: bool = False,
        name: str = "",
    ) -> None:
        super().__init__(name=name)

        if any(
            i != j
            for i, j in zip(binding_mode_0.key(), spacing.binding_mode_key_0)
        ):
            raise ValueError(f"{binding_mode_0} does not match {spacing}")
        if any(
            i != j
            for i, j in zip(binding_mode_1.key(), spacing.binding_mode_key_1)
        ):
            raise ValueError(f"{binding_mode_1} does not match {spacing}")

        # store model attributes
        self._normalize: Tensor = Buffer(torch.tensor(normalize))
        self.spacing = spacing
        self.spacing._binding_cooperativity.add(self)
        self.binding_mode_0 = binding_mode_0
        self.binding_mode_1 = binding_mode_1
        self.binding_mode_0._binding_cooperativity.add(self)
        self.binding_mode_1._binding_cooperativity.add(self)
        self.out_shape_0: int = self.binding_mode_0.out_len(
            self.binding_mode_0.layers[0].input_shape
        )
        self.out_shape_1: int = self.binding_mode_1.out_len(
            self.binding_mode_1.layers[0].input_shape
        )
        self.spacing._update_propagation()

        # store model buffers
        self._bias_bin: Tensor = Buffer(
            torch.tensor(bias_bin, dtype=torch.int32)
        )
        self.train_hill = train_hill
        self.hill = torch.nn.Parameter(
            torch.tensor(0, dtype=__precision__), requires_grad=train_hill
        )

        # create omega
        n_windows_0 = self.binding_mode_0.out_len(
            self.binding_mode_0.layers[0].input_shape
        )
        n_lengths_0 = (
            self.binding_mode_0.out_len(
                self.binding_mode_0.layers[0].max_input_length, "max"
            )
            - self.binding_mode_0.out_len(
                self.binding_mode_0.layers[0].min_input_length, "min"
            )
            + 1
        )
        n_windows_1 = self.binding_mode_1.out_len(
            self.binding_mode_1.layers[0].input_shape
        )
        n_lengths_1 = (
            self.binding_mode_1.out_len(
                self.binding_mode_1.layers[0].max_input_length, "max"
            )
            - self.binding_mode_1.out_len(
                self.binding_mode_1.layers[0].min_input_length, "min"
            )
            + 1
        )
        self.train_omega = train_omega
        self.omega = torch.nn.Parameter(
            torch.zeros(
                size=(
                    n_lengths_0,
                    n_lengths_1,
                    self.spacing.n_strands,
                    self._num_windows(n_windows_0),
                    self._num_windows(n_windows_1),
                ),
                dtype=__precision__,
            ),
            requires_grad=train_omega,
        )

    @property
    def bias_bin(self) -> int:
        return cast(int, self._bias_bin.item())

    @override
    def key(self) -> tuple[Spec]:
        return (self.spacing,)

    @property
    def n_strands(self) -> int:
        return self.spacing.n_strands

    @property
    def normalize(self) -> bool:
        return cast(bool, self._normalize.item())

    @override
    def expected_sequence(self) -> Tensor:
        """Uninformative prior of a sequence"""
        return self.binding_mode_0.expected_sequence()

    @override
    def check_length_consistency(self) -> None:
        for bmd in self.components():
            bmd.check_length_consistency()
        alt_coop = BindingCooperativity(
            spacing=Spacing(
                self.binding_mode_0.key(),
                self.binding_mode_1.key(),
                max_overlap=self.spacing.max_overlap,
                max_spacing=self.spacing.max_spacing,
            ),
            binding_mode_0=self.binding_mode_0,
            binding_mode_1=self.binding_mode_1,
            bias_bin=self.bias_bin,
        )
        # pylint: disable-next=protected-access
        self.binding_mode_0._binding_cooperativity.discard(alt_coop)
        # pylint: disable-next=protected-access
        self.binding_mode_1._binding_cooperativity.discard(alt_coop)
        if self.omega.shape != alt_coop.omega.shape:
            raise RuntimeError(
                f"expected omega shape {alt_coop.omega.shape}"
                f", found {self.omega.shape}"
            )
        if alt_coop.spacing.max_num_windows < self.spacing.max_num_windows:
            raise RuntimeError(
                f"Expected max_num_windows {alt_coop.spacing.max_num_windows}"
                f", found {self.spacing.max_num_windows}"
            )

    @override
    def components(self) -> Iterator[BindingMode]:
        return iter((self.binding_mode_0, self.binding_mode_1))

    @override
    def unfreeze(self, parameter: _unfreezable = "all") -> None:
        """Unfreeze the desired parameter"""
        if parameter in ("hill", "all") and self.train_hill:
            self.hill.requires_grad_()
        if parameter in ("omega", "all") and self.train_omega:
            self.omega.requires_grad_()
        # pylint: disable-next=consider-using-in
        if parameter != "hill" and parameter != "omega":
            super().unfreeze(parameter)

    @override
    def optim_procedure(
        self,
        ancestry: tuple[Component, ...] | None = None,
        current_order: dict[tuple[Spec, ...], BindingOptim] | None = None,
    ) -> dict[tuple[Spec, ...], BindingOptim]:
        if ancestry is None:
            ancestry = tuple()
        if current_order is None:
            current_order = {}
        ancestry = ancestry + (self,)

        # check if already in current_order
        if self.key() not in current_order:
            binding_optim = BindingOptim(
                {ancestry},
                [Step([Call(ancestry[0], "freeze", {})])]
                if len(ancestry) > 0
                else [],
            )
            current_order[self.key()] = binding_optim
        else:
            binding_optim = current_order[self.key()]
            if ancestry in binding_optim.ancestry:
                return current_order
            binding_optim.ancestry.add(ancestry)

        # unfreeze hill
        if self.train_hill:
            binding_optim.steps.append(
                Step([Call(self, "unfreeze", {"parameter": "hill"})])
            )

        # unfreeze scoring parameters if not already being trained
        for bmd in set(
            bmd for bmd in self.components() if bmd.key() not in current_order
        ):
            for layer in bmd.layers:
                layer.update_binding_optim(binding_optim)

        # unfreeze spacing
        self.spacing.update_binding_optim(binding_optim)
        binding_optim.merge_binding_optim()

        # unfreeze omega after every unfreeze spacing
        if self.train_omega:
            spacing_call = Call(
                self.spacing, "unfreeze", {"parameter": "spacing"}
            )
            for step_idx, step in enumerate(binding_optim.steps):
                if spacing_call in step.calls:
                    binding_optim.steps.insert(
                        step_idx + 1,
                        Step([Call(self, "unfreeze", {"parameter": "omega"})]),
                    )

        # unfreeze all parameters
        unfreeze_all = Step(
            [Call(ancestry[0], "unfreeze", {"parameter": "all"})]
        )
        if unfreeze_all not in binding_optim.steps:
            binding_optim.steps.append(unfreeze_all)

        return current_order

    @override
    def max_embedding_size(self) -> int:
        """Maximum number of bytes needed to encode a sequence"""
        return max(
            self.binding_mode_0.max_embedding_size(),
            self.binding_mode_1.max_embedding_size(),
            torch.tensor(data=[], dtype=__precision__).element_size()
            * self.binding_mode_0.out_channels
            * self.binding_mode_0.out_len(self.binding_mode_0.input_shape)
            * self.binding_mode_1.out_channels
            * self.binding_mode_1.out_len(self.binding_mode_0.input_shape),
        )

    def _num_windows(self, input_length: int) -> int:
        """Number of sliding windows modeled by biases"""
        return ceil_div(input_length, self.bias_bin)

    def _update_propagation(
        self,
        binding_mode: BindingMode,
        left_shift: int = 0,
        right_shift: int = 0,
        min_len_shift: int = 0,
        max_len_shift: int = 0,
    ) -> None:
        for bmd_idx in [
            i for i, bmd in enumerate(self.components()) if bmd is binding_mode
        ]:
            left_0 = left_shift if bmd_idx == 0 else 0
            right_0 = right_shift if bmd_idx == 0 else 0
            left_1 = left_shift if bmd_idx == 1 else 0
            right_1 = right_shift if bmd_idx == 1 else 0
            front_0 = -min_len_shift if bmd_idx == 0 else 0
            back_0 = max_len_shift if bmd_idx == 0 else 0
            front_1 = -min_len_shift if bmd_idx == 1 else 0
            back_1 = max_len_shift if bmd_idx == 1 else 0
            self.omega = torch.nn.Parameter(
                F.pad(
                    self.omega,
                    (
                        self._num_windows(self.out_shape_1 + left_1)
                        - self._num_windows(self.out_shape_1),
                        self._num_windows(self.out_shape_1 + left_1 + right_1)
                        - self._num_windows(self.out_shape_1 + left_1),
                        self._num_windows(self.out_shape_0 + left_0)
                        - self._num_windows(self.out_shape_0),
                        self._num_windows(self.out_shape_0 + left_0 + right_0)
                        - self._num_windows(self.out_shape_0 + left_0),
                        0,
                        0,
                        front_1,
                        back_1,
                        front_0,
                        back_0,
                    ),
                ),
                requires_grad=self.omega.requires_grad,
            )
            self.out_shape_0 = self.binding_mode_0.out_len(
                self.binding_mode_0.layers[0].input_shape
            )
            self.out_shape_1 = self.binding_mode_1.out_len(
                self.binding_mode_1.layers[0].input_shape
            )
            # pylint: disable-next=protected-access
            self.spacing._update_propagation()

    def get_spacing(self) -> Tensor:
        n_windows_0 = self.binding_mode_0.out_len(
            self.binding_mode_0.layers[0].input_shape
        )
        n_windows_1 = self.binding_mode_1.out_len(
            self.binding_mode_1.layers[0].input_shape
        )

        # create diagonal-constant matrix for each strand
        spacing = self.spacing.get_spacing(
            n_windows_0=n_windows_0, n_windows_1=n_windows_1
        )

        if not self.omega.requires_grad and not torch.any(self.omega != 0):
            return spacing.unsqueeze(0)

        # pad omega to the right shape
        omega: Tensor = self.omega
        if self.normalize:
            omega = omega - omega.mean(dim=(-1, -2, -3), keepdim=True)
        omega = omega.repeat_interleave(self.bias_bin, -2).repeat_interleave(
            self.bias_bin, -1
        )[..., :n_windows_0, :n_windows_1]
        if self.spacing.n_strands == 1:
            omega.squeeze_(-3)
        else:
            omegas = omega.unbind(-3)
            omega = torch.cat(
                (
                    torch.cat((omegas[0], omegas[1]), dim=-1),
                    torch.cat((omegas[0], omegas[1]), dim=-1),
                ),
                dim=-2,
            )
        assert spacing.shape == omega.shape[-2:]

        # return spacing + omega padded for length indexing
        min_windows_0 = self.binding_mode_0.out_len(
            self.binding_mode_0.layers[0].min_input_length, "min"
        )
        min_windows_1 = self.binding_mode_1.out_len(
            self.binding_mode_1.layers[0].min_input_length, "min"
        )
        return F.pad(
            spacing + omega, (0, 0, 0, 0, min_windows_1, 0, min_windows_0, 0)
        )

    @override
    @Transform.cache
    def score_windows(self, seqs: Tensor) -> Tensor:
        spacing = self.get_spacing()

        windows_0 = self.binding_mode_0.score_windows(seqs)
        windows_1 = self.binding_mode_1.score_windows(seqs)

        if spacing.shape[0] > 1:
            with torch.no_grad():
                n_lengths_0 = windows_0[:, 0].isfinite().sum(dim=-1)
                n_lengths_1 = windows_1[:, 0].isfinite().sum(dim=-1)
            spacing = spacing[n_lengths_0, n_lengths_1]

        return (
            windows_0.flatten(1, 2).unsqueeze(2)
            + windows_1.flatten(1, 2).unsqueeze(1)
        ) + spacing

    @override
    @Transform.cache
    def forward(self, seqs: Tensor) -> Tensor:
        """Score a batch-first tensor of sequences"""
        return (torch.exp(self.hill) * self.score_windows(seqs)).logsumexp(
            (-1, -2)
        )
