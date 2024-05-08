"""Cooperativity between two binding modes.

Members are explicitly re-exported in pyprobound.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self, override

from . import __precision__
from .base import Binding, BindingOptim, Call, Component, Spec, Step, Transform
from .containers import TParameterList
from .layers import LayerSpec, ModeKey
from .mode import Mode
from .utils import ceil_div


class Spacing(Spec):
    r"""Experiment-independent dimer cooperativity modeling.

    :math:`\omega_{a:b}(x^a, x^b)` represents a scaling factor to the outer sum
    of sliding windows of the two binding modes in the dimer. Its experiment-
    independent formulation consists of a scaling for each `relative` offset
    between any pair of sliding windows, such that
    :math:`\omega_{a:b}(x^a_m, x^b_n)
    = \omega_{a:b}(x^a_{m+i}, x^b_{n+i}) \forall i`.

    Attributes:
        mode_key_a (ModeKey): The specification of the first binding
            mode contributing to the dimer.
        mode_key_b (ModeKey): The specification of the second binding
            mode contributing to the dimer.
        log_spacing (Tensor): The flattened representation of
            :math:`\omega_{a:b}(x^a, x^b)`.
    """

    unfreezable = Literal[Spec.unfreezable, "spacing"]

    def __init__(
        self,
        mode_key_a: ModeKey,
        mode_key_b: ModeKey,
        score_same: bool = True,
        score_reverse: bool = True,
        score_mode: Literal["both", "positive", "negative"] = "both",
        max_overlap: int | None = None,
        max_spacing: int | None = None,
        normalize: bool = False,
        name: str = "",
    ) -> None:
        r"""Initializes the experiment-independent dimer cooperativity.

        Args:
            mode_key_a: The specification of the first binding mode
                contributing to the dimer.
            mode_key_b: The specification of the second binding mode
                contributing to the dimer.
            max_overlap: The maximum number of bases shared by two windows.
            max_spacing: The maximum number of bases apart two windows can be.
            normalize: Whether to mean-center `log_spacing` over all windows.
            name: A string used to describe the cooperativity.
        """
        super().__init__(name=name)

        self.mode_key_a = mode_key_a
        self.mode_key_b = mode_key_b
        self.log_spacing = torch.nn.Parameter(
            torch.zeros(size=(self.n_strands, 1), dtype=__precision__)
        )
        self.max_overlap = max_overlap
        self.max_spacing = max_spacing
        self.normalize = normalize
        self._cooperativities: set[Cooperativity] = set()
        self._score_same = score_same
        self._score_reverse = score_reverse
        self._score_mode = score_mode

    @classmethod
    def from_specs(
        cls,
        specs_a: Iterable[LayerSpec],
        specs_b: Iterable[LayerSpec],
        score_same: bool = True,
        score_reverse: bool = True,
        score_mode: Literal["both", "positive", "negative"] = "both",
        max_overlap: int | None = None,
        max_spacing: int | None = None,
        normalize: bool = False,
        name: str = "",
    ) -> Self:
        r"""Creates a new the experiment-independent dimer cooperativity.

        Args:
            specs_a: The specifications of the first binding mode
                contributing to the dimer.
            specs_b: The specification of the second binding mode
                contributing to the dimer.
            max_overlap: The maximum number of bases shared by two windows.
            max_spacing: The maximum number of bases apart two windows can be.
            normalize: Whether to mean-center `log_spacing` over all
                windows.
            name: A string used to describe the cooperativity.
        """
        return cls(
            ModeKey(specs_a),
            ModeKey(specs_b),
            score_same=score_same,
            score_reverse=score_reverse,
            score_mode=score_mode,
            max_overlap=max_overlap,
            max_spacing=max_spacing,
            normalize=normalize,
            name=name,
        )

    @property
    def n_strands(self) -> Literal[1, 2]:
        """The number of output channels of the component modes."""
        if (
            self.mode_key_a[-1].out_channels not in (1, 2)
            or self.mode_key_a[-1].out_channels
            != self.mode_key_b[-1].out_channels
        ):
            raise RuntimeError(
                f"{self.mode_key_a} has"
                f" {self.mode_key_a[-1].out_channels} strands"
                f" but {self.mode_key_b} has"
                f" {self.mode_key_b[-1].out_channels} strands"
            )
        return cast(Literal[1, 2], self.mode_key_a[-1].out_channels)

    @property
    def max_num_windows(self) -> int:
        """The number of sliding windows modeled by `log_spacing`."""
        return (self.log_spacing.shape[-1] + 1) // 2

    @override
    def components(self) -> Iterator[Component]:
        return iter(())

    @override
    def unfreeze(self, parameter: unfreezable = "all") -> None:
        if parameter in ("spacing", "all"):
            self.log_spacing.requires_grad_()
        if parameter != "spacing":
            super().unfreeze(parameter)

    @override
    def update_binding_optim(
        self, binding_optim: BindingOptim
    ) -> BindingOptim:
        # Unfreeze spacing with first monomer of Conv1d if available
        insertion_idx = len(binding_optim.steps)
        append = False
        for step_idx, step in enumerate(binding_optim.steps):
            for call in step.calls:
                if call.fun == "activity_heuristic":
                    insertion_idx = step_idx + 1
                if call.fun == "unfreeze":
                    if call.kwargs["parameter"] in ("monomer", "spacing"):
                        insertion_idx = min(step_idx, insertion_idx)
                        append = True

        new_call = Call(self, "unfreeze", {"parameter": "spacing"})
        if append:
            binding_optim.steps[insertion_idx].calls.append(new_call)
        else:
            binding_optim.steps.insert(insertion_idx, Step([new_call]))

        return binding_optim

    def _update_propagation(self) -> None:
        """Updates the number of windows, called by parent Cooperativity."""
        new_max_num_windows = max(
            max(coop.n_windows_a, coop.n_windows_b)
            for coop in self._cooperativities
        )
        if new_max_num_windows < 1:
            raise RuntimeError("Cannot shrink spacing to less than 1 window")
        shift = new_max_num_windows - self.max_num_windows
        self.log_spacing = torch.nn.Parameter(
            F.pad(self.log_spacing, (shift, shift)),
            requires_grad=self.log_spacing.requires_grad,
        )

    def get_log_spacing(self, n_windows_a: int, n_windows_b: int) -> Tensor:
        r"""Adjusts log_spacing for windows, max_spacing, and max_overlap.

        Args:
            n_windows_a: The number of windows in the first binding mode.
            n_windows_b: The number of windows in the second binding mode.

        Returns:
            A tensor with the flattened representation of the cooperativity
            of each pair of windows :math:`(x^a_m, x^b_n)`, of shape
            :math:`(\text{out_channels},
            \text{out_length}_a+\text{out_length}_b-1)`.
        """
        size_a = self.mode_key_a.in_len(1, mode="max")
        size_b = self.mode_key_b.in_len(1, mode="max")
        if size_a is None or size_b is None:
            raise RuntimeError(
                f"Receptive field of {self} is calculated as"
                f" {size_a} and {size_b}; expected int for both"
            )

        log_spacing: Tensor = self.log_spacing

        # Adjust for max_overlap and max_spacing
        # overlap(offset) = min(
        #   min(size_a, size_b),
        #   size_a - offset if offset > 0 else size_b + offset
        # )
        # spacing = -overlap(offset)
        if self.max_spacing is not None:
            log_spacing = log_spacing.index_fill(  # positive offsets
                -1,
                torch.arange(
                    self.max_num_windows + self.max_spacing + size_a,
                    log_spacing.shape[-1],
                ),
                float("-inf"),
            )
            log_spacing = log_spacing.index_fill(  # negative offsets
                -1,
                torch.arange(
                    0, self.max_num_windows - self.max_spacing - size_b - 1
                ),
                float("-inf"),
            )
        if self.max_overlap is not None and self.max_overlap < min(
            size_a, size_b
        ):
            log_spacing = log_spacing.index_fill(
                -1,
                torch.arange(
                    self.max_num_windows - size_b + self.max_overlap,
                    self.max_num_windows + size_a - self.max_overlap - 1,
                ),
                float("-inf"),
            )
        if self._score_mode != "both":
            midpoint = self.max_num_windows + ((size_a - size_b) // 2)
            log_spacing = log_spacing.index_fill(
                -1,
                (
                    torch.arange(0, midpoint)
                    if self._score_mode == "positive"
                    else torch.arange(midpoint, log_spacing.shape[-1])
                ),
                float("-inf"),
            )

        # Adjust strands
        if not self._score_same:
            log_spacing = log_spacing.index_fill(
                0, torch.tensor(0), float("-inf")
            )
        if not self._score_reverse:
            log_spacing = log_spacing.index_fill(
                0, torch.tensor(1), float("-inf")
            )

        # Shift to the right index
        shift_left = self.max_num_windows - n_windows_a
        shift_right = n_windows_b - self.max_num_windows
        if shift_right == 0:
            log_spacing = log_spacing[:, shift_left:]
        else:
            log_spacing = log_spacing[:, shift_left:shift_right]
        assert log_spacing.shape == (
            self.n_strands,
            n_windows_a + n_windows_b - 1,
        )

        if self.normalize:
            log_spacing = (
                log_spacing
                - log_spacing.nan_to_num(
                    posinf=float("nan"), neginf=float("nan")
                ).nanmean()
            )

        return log_spacing

    def get_log_spacing_matrix(
        self, n_windows_a: int, n_windows_b: int
    ) -> Tensor:
        r"""The diagonal-constant cooperativity :math:`\omega_{a:b}(x^a, x^b)`.

        Args:
            n_windows_a: The number of windows in the first binding mode.
            n_windows_b: The number of windows in the second binding mode.

        Returns:
            A diagonal-constant tensor with the cooperativity of each pair of
            windows :math:`(x^a_m, x^b_n)` of shape
            :math:`(\text{n_strands}_a,\text{n_strands}_b,
            \text{out_length}_a,\text{out_length}_b)`.)`
        """
        log_spacing = self.get_log_spacing(n_windows_a, n_windows_b)

        # Convert to diagonal constant matrix
        log_spacing = log_spacing.unfold(-1, n_windows_b, 1).flip(-2)
        assert log_spacing.shape == (self.n_strands, n_windows_a, n_windows_b)

        # Stack strands into a single matrix
        if self.n_strands == 2:
            log_spacing = torch.stack(
                (log_spacing, log_spacing.flip(-1, -2, -3)), 1
            )
        else:
            log_spacing.unsqueeze_(0)
        assert log_spacing.shape == (
            self.n_strands,
            self.n_strands,
            n_windows_a,
            n_windows_b,
        )

        return log_spacing


class Cooperativity(Binding):
    r"""Experiment-specific dimer cooperativity modeling with position bias.

    .. math::
        \log \frac{\omega_{a:b}(x^a, x^b)}{
            K^{rel}_{\text{D}, a} (S_{i, x^a})
            K^{rel}_{\text{D}, b} (S_{i, x^b})
        }

    Attributes:
        spacing (Spacing): The experiment-independent specification of the
            dimer.
        mode_a (Mode): The first binding mode contributing to the dimer.
        mode_b (Mode): The second binding mode contributing to the
            dimer.
        log_hill (Tensor): The Hill coeffient in log space.
        log_posbias (list[Tensor]): The bias :math:`\omega_{a:b}(x^a_m, x^b_n)`
            at each relative offset :math:`n-m` for each strand.
    """

    unfreezable = Literal[Binding.unfreezable, "hill", "posbias"]
    _cache_fun = "score_windows"

    def __init__(
        self,
        spacing: Spacing,
        mode_a: Mode,
        mode_b: Mode,
        train_hill: bool = False,
        train_posbias: bool = False,
        bias_mode: Literal["strand", "same", "reverse"] = "strand",
        bias_bin: int = 1,
        length_specific_bias: bool = True,
        normalize: bool = False,
        name: str = "",
    ) -> None:
        r"""Initializes the experiment-specific dimer cooperativity.

        Args:
            spacing: The experiment-independent specification of the dimer.
            mode_a: The first binding mode contributing to the dimer.
            mode_b: The second binding mode contributing to the dimer.
            train_hill: Whether to train a Hill coefficient for the dimer.
            train_posbias: Whether to train posbias parameter
                :math:`\omega_{a:b}(x^a, x^b)` for each pair of windows
                :math:`(x^a_m, x^b_n)`.
            bias_mode: Whether to train a separate bias for each strand, use
                the same bias across both strands, or a reversed bias for the
                opposite strand.
            bias_bin: Applies constraint :math:`\omega_{a:b}(
                x^a_{i\times\text{bias_bin}}, x^b_{j\times\text{bias_bin}}
                ) =` :math:`\cdots = \omega_{a:b}(
                x^a_{(i+1)\times\text{bias_bin}-1},
                x^b_{(j+1)\times\text{bias_bin}-1}
                )`.
            length_specific_bias: Whether to train a separate bias parameter
                for each input length.
            normalize: Whether to mean-center `log_posbias` over all windows.
            name: A string used to describe the cooperativity.
        """
        super().__init__(name=name)

        if any(i != j for i, j in zip(mode_a.key(), spacing.mode_key_a)):
            raise ValueError(f"{mode_a} does not match {spacing}")
        if any(i != j for i, j in zip(mode_b.key(), spacing.mode_key_b)):
            raise ValueError(f"{mode_b} does not match {spacing}")

        # Store model attributes
        self.normalize = normalize
        self.spacing = spacing
        self.spacing._cooperativities.add(self)
        self.mode_a = mode_a
        self.mode_b = mode_b
        self.mode_a._cooperativities.add(self)
        self.mode_b._cooperativities.add(self)
        self.n_windows_a: int = self.mode_a.out_len(self.mode_a.input_shape)
        self.n_windows_b: int = self.mode_b.out_len(self.mode_b.input_shape)
        self._bias_mode = bias_mode
        self._bias_bin = bias_bin
        self._length_specific_bias = length_specific_bias
        self.train_hill = train_hill
        self.log_hill = torch.nn.Parameter(
            torch.tensor(0, dtype=__precision__), requires_grad=train_hill
        )
        self.spacing._update_propagation()

        # Create posbias
        lengths_a = (
            (
                self.mode_a.out_len(self.mode_a.max_input_length, "max")
                - self.mode_a.out_len(self.mode_a.min_input_length, "min")
                + 1
            )
            if self.length_specific_bias
            else 1
        )
        lengths_b = (
            (
                self.mode_b.out_len(self.mode_b.max_input_length, "max")
                - self.mode_b.out_len(self.mode_b.min_input_length, "min")
                + 1
            )
            if self.length_specific_bias
            else 1
        )
        self.train_posbias = train_posbias
        self.log_posbias = TParameterList(
            [
                torch.zeros(
                    lengths_a,
                    lengths_b,
                    self.spacing.n_strands,
                    (self.spacing.n_strands if bias_mode == "strand" else 1),
                    self._num_windows(
                        min(i, self.n_windows_a, self.n_windows_b)
                    ),
                )
                for i in itertools.chain(
                    range(1, self.n_windows_a + 1),
                    range(self.n_windows_b - 1, 0, -1),
                )
            ]
        )
        self.log_posbias.requires_grad_(train_posbias)

    @property
    def n_strands(self) -> Literal[1, 2]:
        """The number of output channels of the component modes."""
        return self.spacing.n_strands

    @property
    def bias_mode(self) -> Literal["strand", "same", "reverse"]:
        """Whether to train a separate bias for each strand, use the same bias
        across both strands, or a reversed bias for the opposite strand."""
        return self._bias_mode

    @property
    def bias_bin(self) -> int:
        r"""Applies the constraint
        :math:`\omega_{a:b}(
        x^a_{i\times\text{bias_bin}}, x^b_{j\times\text{bias_bin}}
        ) =` :math:`\cdots = \omega_{a:b}(
        x^a_{(i+1)\times\text{bias_bin}-1}, x^b_{(j+1)\times\text{bias_bin}-1}
        )`.
        """
        return self._bias_bin

    @property
    def length_specific_bias(self) -> bool:
        """Whether to train a separate bias parameter for each input length."""
        return self._length_specific_bias

    def _num_windows(self, input_length: int) -> int:
        """The number of sliding windows modeled by biases."""
        return ceil_div(input_length, self.bias_bin)

    @override
    def key(self) -> tuple[Spec]:
        return (self.spacing,)

    @override
    def components(self) -> Iterator[Mode]:
        return iter((self.mode_a, self.mode_b))

    @override
    def max_embedding_size(self) -> int:
        return 3 * min(
            self.mode_a.max_embedding_size(), self.mode_b.max_embedding_size()
        )

    @override
    def check_length_consistency(self) -> None:
        for bmd in self.components():
            bmd.check_length_consistency()
        alt_coop = Cooperativity(
            spacing=Spacing(
                self.mode_a.key(),
                self.mode_b.key(),
                max_overlap=self.spacing.max_overlap,
                max_spacing=self.spacing.max_spacing,
            ),
            mode_a=self.mode_a,
            mode_b=self.mode_b,
            bias_mode=self.bias_mode,
            bias_bin=self.bias_bin,
            length_specific_bias=self.length_specific_bias,
        )
        # pylint: disable-next=protected-access
        self.mode_a._cooperativities.discard(alt_coop)
        # pylint: disable-next=protected-access
        self.mode_b._cooperativities.discard(alt_coop)
        if (
            alt_coop.n_windows_a != self.n_windows_a
            or alt_coop.n_windows_b != self.n_windows_b
        ):
            raise RuntimeError(
                "Expected number of windows per mode",
                f" {(alt_coop.n_windows_a, alt_coop.n_windows_b)}",
                f", found {(self.n_windows_a, self.n_windows_b)}",
            )
        for i, (log_posbias, alt_log_posbias) in enumerate(
            zip(self.log_posbias, alt_coop.log_posbias)
        ):
            if log_posbias.shape != alt_log_posbias.shape:
                raise RuntimeError(
                    f"expected posbias shape {alt_log_posbias.shape}"
                    f" for index {i}, found {log_posbias.shape}"
                )

    @override
    def unfreeze(self, parameter: unfreezable = "all") -> None:
        if parameter in ("hill", "all") and self.train_hill:
            self.log_hill.requires_grad_()
        if parameter in ("posbias", "all") and self.train_posbias:
            self.log_posbias.requires_grad_()
        # pylint: disable-next=consider-using-in
        if parameter != "hill" and parameter != "posbias":
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

        # Check if already in current_order
        if self.key() not in current_order:
            binding_optim = BindingOptim(
                {ancestry},
                (
                    [Step([Call(ancestry[0], "freeze", {})])]
                    if len(ancestry) > 0
                    else []
                ),
            )
            current_order[self.key()] = binding_optim
        else:
            binding_optim = current_order[self.key()]
            if ancestry in binding_optim.ancestry:
                return current_order
            binding_optim.ancestry.add(ancestry)

        # Unfreeze scoring parameters if not already being trained
        for bmd in set(
            bmd for bmd in self.components() if bmd.key() not in current_order
        ):
            for layer in bmd.layers:
                layer.update_binding_optim(binding_optim)

        # Unfreeze spacing
        self.spacing.update_binding_optim(binding_optim)
        binding_optim.merge_binding_optim()

        # Unfreeze posbias after every unfreeze spacing
        if self.train_posbias:
            spacing_call = Call(
                self.spacing, "unfreeze", {"parameter": "spacing"}
            )
            for step_idx, step in enumerate(binding_optim.steps):
                if spacing_call in step.calls:
                    binding_optim.steps.insert(
                        step_idx + 1,
                        Step(
                            [Call(self, "unfreeze", {"parameter": "posbias"})]
                        ),
                    )

        # Unfreeze all parameters
        unfreeze_all = Step(
            [Call(ancestry[0], "unfreeze", {"parameter": "all"})]
        )
        if unfreeze_all not in binding_optim.steps:
            binding_optim.steps.append(unfreeze_all)

        return current_order

    def _update_propagation(
        self,
        binding_mode: Mode,
        left_shift: int = 0,
        right_shift: int = 0,
        min_len_shift: int = 0,
        max_len_shift: int = 0,
    ) -> None:
        """Updates the input shape, called by a child Mode after its update.

        Args:
            binding_mode: The child component requesting the update.
            left_shift: The change in size on the left side of the sequence.
            right_shift: The change in size on the right side of the sequence.
            min_len_shift: The change in the number of short input lengths.
            max_len_shift: The change in the number of long input lengths.
        """

        def update(
            old_lengths: list[int],
            direction: Literal["left_a", "left_b", "right_a", "right_b"],
            sign: Literal[-1, 1],
        ) -> list[int]:
            # Update number of diagonals
            if direction in ("left_a", "right_b"):  # append
                if sign > 0:
                    self.log_posbias.append(
                        torch.nn.Parameter(
                            torch.zeros(
                                self.log_posbias[0].shape[:-1] + (0,),
                                dtype=__precision__,
                            ),
                            requires_grad=self.train_posbias,
                        )
                    )
                    old_lengths = old_lengths + [0]
                else:
                    self.log_posbias = self.log_posbias[:-1]
                    old_lengths = old_lengths[:-1]
            else:  # prepend
                if sign > 0:
                    self.log_posbias.insert(
                        0,
                        torch.nn.Parameter(
                            torch.zeros(
                                self.log_posbias[0].shape[:-1] + (0,),
                                dtype=__precision__,
                            ),
                            requires_grad=self.train_posbias,
                        ),
                    )
                    old_lengths = [0] + old_lengths
                else:
                    self.log_posbias = self.log_posbias[1:]
                    old_lengths = old_lengths[1:]

            if direction in ("left_a", "right_a"):
                self.n_windows_a += sign
            else:
                self.n_windows_b += sign

            # Update diagonal lengths
            for diag_idx, diag in enumerate(self.log_posbias):
                prepend = 0
                append = 0
                shift = self._num_windows(
                    old_lengths[diag_idx] + sign
                ) - self._num_windows(old_lengths[diag_idx])
                increment = False

                if direction == "left_a" and (
                    (sign > 0 and diag_idx >= self.n_windows_a - 1)
                    or (sign < 0 and diag_idx >= self.n_windows_a)
                ):
                    prepend = shift
                    increment = True
                elif direction == "left_b" and (
                    (sign > 0 and diag_idx < self.n_windows_a)
                    or (sign < 0 and diag_idx < self.n_windows_a - 1)
                ):
                    prepend = shift
                    increment = True
                elif direction == "right_a" and (
                    (sign > 0 and diag_idx < self.n_windows_b)
                    or (sign < 0 and diag_idx < self.n_windows_b - 1)
                ):
                    append = shift
                    increment = True
                elif direction == "right_b" and (
                    (sign > 0 and diag_idx >= self.n_windows_b - 1)
                    or (sign < 0 and diag_idx >= self.n_windows_b)
                ):
                    append = shift
                    increment = True
                self.log_posbias[diag_idx] = torch.nn.Parameter(
                    F.pad(diag, (prepend, append)),
                    requires_grad=diag.requires_grad,
                )
                if increment:
                    old_lengths[diag_idx] = min(
                        old_lengths[diag_idx] + sign,
                        self.n_windows_a,
                        self.n_windows_b,
                    )

            return old_lengths

        for bmd_idx in [
            i for i, bmd in enumerate(self.components()) if bmd is binding_mode
        ]:
            left_a = left_shift if bmd_idx == 0 else 0
            right_a = right_shift if bmd_idx == 0 else 0
            left_b = left_shift if bmd_idx == 1 else 0
            right_b = right_shift if bmd_idx == 1 else 0
            front_a = -min_len_shift if bmd_idx == 0 else 0
            back_a = max_len_shift if bmd_idx == 0 else 0
            front_b = -min_len_shift if bmd_idx == 1 else 0
            back_b = max_len_shift if bmd_idx == 1 else 0
            if not self.length_specific_bias:
                front_a, front_b, back_a, back_b = 0, 0, 0, 0

            # Run updates
            old_lengths = [
                min(i, self.n_windows_a, self.n_windows_b)
                for i in itertools.chain(
                    range(1, self.n_windows_a + 1),
                    range(self.n_windows_b - 1, 0, -1),
                )
            ]
            if left_a != 0:
                for _ in range(abs(left_a)):
                    old_lengths = update(
                        old_lengths, "left_a", 1 if left_a > 0 else -1
                    )
            if left_b != 0:
                for _ in range(abs(left_b)):
                    old_lengths = update(
                        old_lengths, "left_b", 1 if left_b > 0 else -1
                    )
            if right_a != 0:
                for _ in range(abs(right_a)):
                    old_lengths = update(
                        old_lengths, "right_a", 1 if right_a > 0 else -1
                    )
            if right_b != 0:
                for _ in range(abs(right_b)):
                    old_lengths = update(
                        old_lengths, "right_b", 1 if right_b > 0 else -1
                    )

            # Update input lengths
            for diag_idx, diagonal in enumerate(self.log_posbias):
                self.log_posbias[diag_idx] = torch.nn.Parameter(
                    F.pad(
                        diagonal,
                        (0, 0, 0, 0, 0, 0, front_b, back_b, front_a, back_a),
                    ),
                    requires_grad=diagonal.requires_grad,
                )

            # pylint: disable-next=protected-access
            self.spacing._update_propagation()

    def get_log_spacing(self) -> Tensor:
        r"""The flattened cooperativity :math:`\omega_{a:b}(x^a, x^b)`.

        Returns:
            A tensor with the flattened representation of the cooperativity
            of each pair of windows :math:`(x^a_m, x^b_n)`, of shape
            :math:`(\text{n_strands},
            \text{out_length}_a+\text{out_length}_b-1)`.
        """
        return self.spacing.get_log_spacing(
            n_windows_a=self.n_windows_a, n_windows_b=self.n_windows_b
        )

    def get_log_spacing_diagonals(self) -> list[Tensor]:
        r"""The cooperativity position bias :math:`\omega_{a:b}(x^a, x^b)`.

        Returns:
            A list of tensors with the bias of each pair of windows
            :math:`(x^a_m, x^b_n)` at each relative offset :math:`n-m`, each of
            shape :math:`(\text{input_lengths}_a,\text{input_lengths}_b,
            \text{n_strands}_a,\text{n_strands}_b,\text{diagonal_length})`.)`.
        """

        # Normalize posbias
        log_posbias: list[Tensor] = list(self.log_posbias)
        if self.normalize:
            for i, diagonal in enumerate(log_posbias):
                log_posbias[i] = diagonal - diagonal.mean(dim=-1, keepdim=True)

        # Pad posbias to the right shape
        if self.bias_bin > 1:
            for i, length in enumerate(
                itertools.chain(
                    range(1, self.n_windows_a + 1),
                    range(self.n_windows_b - 1, 0, -1),
                )
            ):
                log_posbias[i] = log_posbias[i].repeat_interleave(
                    self.bias_bin, -1
                )[..., : min(length, self.n_windows_a, self.n_windows_b)]

        # Adjust for the opposite strand
        if self.bias_mode != "strand":
            flip = [-3]
            if self.bias_mode == "reverse":
                flip.extend([-2, -1])
            for i, diagonal in enumerate(log_posbias):
                diagonal = diagonal[..., :1, :]
                log_posbias[i] = torch.cat(
                    (diagonal, diagonal.flip(flip)), dim=-2
                )

        # Add spacing value and return
        log_spacing = self.get_log_spacing()
        if self.n_strands > 1:
            log_spacing = torch.stack(
                [log_spacing, log_spacing.flip(-1, -2)], -2
            )
        log_spacing_diagonals = [
            l_s.unsqueeze(-1) + diagonal
            for l_s, diagonal in zip(log_spacing.unbind(-1), log_posbias)
        ]

        # Return spacing + posbias padded for length indexing
        if self.length_specific_bias:
            min_a = self.mode_a.out_len(self.mode_a.min_input_length, "min")
            min_b = self.mode_b.out_len(self.mode_b.min_input_length, "min")
            for i, diag in enumerate(log_spacing_diagonals):
                log_spacing_diagonals[i] = F.pad(
                    diag, (0, 0, 0, 0, 0, 0, min_b, 0, min_a, 0)
                )
        return log_spacing_diagonals

    def get_log_spacing_matrix(self) -> Tensor:
        r"""The cooperativity position bias :math:`\omega_{a:b}(x^a, x^b)`.

        Returns:
            A tensor with the bias of each pair of windows
            :math:`(x^a_m, x^b_n)` of shape
            :math:`(\text{input_lengths}_a,\text{input_lengths}_b,
            \text{n_strands}_a,\text{n_strands}_b,
            \text{out_length}_a,\text{out_length}_b)`.)`.
        """
        if not any(
            diag.requires_grad or torch.any(diag != 0)
            for diag in self.log_posbias
        ):
            return (
                self.spacing.get_log_spacing_matrix(
                    n_windows_a=self.n_windows_a, n_windows_b=self.n_windows_b
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

        log_spacing_diagonals = self.get_log_spacing_diagonals()

        # Form into matrix
        log_spacing_matrix = torch.zeros(
            list(log_spacing_diagonals[0].shape[:-1])
            + [self.n_windows_a, self.n_windows_b]
        )
        for diagonal, offset in zip(
            log_spacing_diagonals,
            range(-self.n_windows_a + 1, self.n_windows_b),
        ):
            log_spacing_matrix = log_spacing_matrix.diagonal_scatter(
                diagonal, offset, -2, -1
            )

        return log_spacing_matrix

    @override
    def expected_sequence(self) -> Tensor:
        return self.mode_a.expected_sequence()

    @override
    @Transform.cache
    def score_windows(self, seqs: Tensor) -> Tensor:
        r"""Calculates the log score of each window before summing over them.

        .. math::
            \log \frac{\omega_{a:b}(x^a, x^b)}{
                K^{rel}_{\text{D}, a} (S_{i, x^a})
                K^{rel}_{\text{D}, b} (S_{i, x^b})
            }

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{in_length})`.

        Returns:
            A tensor with the score of each window of shape
            :math:`(\text{minibatch},\text{n_strands}_a,\text{n_strands}_b,
            \text{out_length}_a,\text{out_length}_b)`.
        """
        log_spacing = self.get_log_spacing_matrix()
        windows_a = self.mode_a.score_windows(seqs)
        windows_b = self.mode_b.score_windows(seqs)

        if len(log_spacing) > 1:
            with torch.no_grad():
                lengths_a = windows_a[:, 0].isfinite().sum(dim=-1)
                lengths_b = windows_b[:, 0].isfinite().sum(dim=-1)
            log_spacing = log_spacing[lengths_a, lengths_b]
        else:
            log_spacing = log_spacing[0, 0]

        return log_spacing + (
            windows_a.unsqueeze(-2).unsqueeze(-1)
            + windows_b.unsqueeze(-3).unsqueeze(-2)
        )

    @override
    @Transform.cache
    def forward(self, seqs: Tensor) -> Tensor:
        r"""Calculates the log score of each sequence.

        .. math::
            \log \frac{1}{
                K^{rel}_{\text{D}, a} (S_i)
                K^{rel}_{\text{D}, b} (S_i)
            } = \log \sum_{x^a, x^b} \frac{\omega_{a:b}(x^a, x^b)}{
                K^{rel}_{\text{D}, a} (S_{i, x^a})
                K^{rel}_{\text{D}, b} (S_{i, x^b})
            }

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            The log score tensor of shape :math:`(\text{minibatch},)`.
        """
        out = torch.full(
            (len(seqs),),
            float("-inf"),
            dtype=__precision__,
            device=self.log_hill.device,
        )

        # Score each mode
        windows_a = self.mode_a.score_windows(seqs)
        windows_b = self.mode_b.score_windows(seqs)

        # Get spacing
        posbias = any(
            diag.requires_grad or torch.any(diag != 0)
            for diag in self.log_posbias
        )
        if posbias:
            log_spacing_diagonals = self.get_log_spacing_diagonals()
            if len(log_spacing_diagonals[0]) > 1:
                with torch.no_grad():
                    lengths_a = windows_a[:, 0].isfinite().sum(dim=-1)
                    lengths_b = windows_b[:, 0].isfinite().sum(dim=-1)
            else:
                log_spacing_diagonals = [
                    i.squeeze(0) for i in log_spacing_diagonals
                ]
        else:
            log_spacing_vector = self.get_log_spacing()

        # Loop over relative offsets
        for i in range(self.n_windows_a + self.n_windows_b - 1):
            # Get relative offset
            if posbias:
                log_spacing = log_spacing_diagonals[i]
                if len(log_spacing) > 1:
                    log_spacing = log_spacing[lengths_a, lengths_b]
            else:
                log_spacing = log_spacing_vector[..., i]
            if torch.all(torch.isneginf(log_spacing)):
                continue

            # Get slices of offsets
            left = -i - 1
            right = self.n_windows_a + self.n_windows_b - 1 - i
            sl_0 = slice(left, right)
            sl_1 = slice(-right, -left)

            # Loop over strands
            for st_a in range(self.n_strands):
                for st_b in range(self.n_strands):
                    if posbias:
                        weight = log_spacing[..., st_a, st_b, :]
                        if torch.all(torch.isneginf(weight)):
                            continue
                        diagonal = (
                            weight
                            + windows_a[:, st_a, sl_0]
                            + windows_b[:, st_b, sl_1]
                        )
                    else:
                        weight = log_spacing[0 if st_a == st_b else -1]
                        if torch.all(torch.isneginf(weight)):
                            continue
                        diagonal = (
                            weight
                            + windows_a[:, st_a, sl_0 if st_b == 0 else sl_1]
                            + windows_b[:, st_b, sl_1 if st_b == 0 else sl_0]
                        )
                    out = torch.logaddexp(
                        out,
                        (torch.exp(self.log_hill) * diagonal).logsumexp(-1),
                    )

        return out
