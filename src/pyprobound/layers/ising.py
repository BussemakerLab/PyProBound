r"""Ising model of cooperative nucleotide recognition.

Members are explicitly re-exported in pyprobound.layers.

Implements the model of Schwager et al. In vitro binding energies capture Klf4
occupancy across the human genome. https://doi.org/10.48550/arXiv.2601.16151
"""

from __future__ import annotations

import itertools
from typing import Any, Literal

import torch
from torch import Tensor
from typing_extensions import override

from ..aggregate import Contribution
from ..base import BindingOptim, Call, Step
from .conv1d import Conv1d
from .psam import PSAM


class IsingPSAM(PSAM):
    r"""PSAM with an Ising coupling between adjacent positions.

    Each position of a window is in one of two states: :math:`\sigma_i = +1`
    recognizes the nucleotide, contributing :math:`\beta_\phi`, and
    :math:`\sigma_i = -1` is a sequence-independent alternative state
    contributing nothing. Adjacent positions are coupled by :math:`J_i`.

    .. math::
        \log \frac{1}{K^{rel}_{\text{D}, a} (S_{i, x})}
        = \log \sum_{\sigma} \exp \left[
            \sum_i J_i (\sigma_i \sigma_{i+1} - 1)
            + \sum_i \tfrac{\sigma_i + 1}{2} \beta_{\phi_i}
        \right]

    The gauge :math:`J_i (\sigma_i \sigma_{i+1} - 1)` gives the
    all-:math:`\sigma = -1` state a weight of exactly 1 regardless of
    :math:`J`, so stepping `coupling_schedule` does not shift :math:`E_0`.

    Attributes:
        coupling (Tensor): The coupling :math:`J`, one element per group.
        coupling_groups (Tensor): The group index of each of the
            `kernel_size - 1` nucleotide interfaces.
    """

    unfreezable = Literal[PSAM.unfreezable, "coupling"]

    def __init__(
        self,
        *args: Any,
        coupling: float | Tensor = 0.0,
        train_coupling: bool = True,
        coupling_schedule: tuple[float, ...] = (0.0, 1.0, 2.1),
        coupling_with_monomer: bool = False,
        coupling_groups: Tensor | None = None,
        method: Literal["enumerate", "transfer"] = "transfer",
        **kwargs: Any,
    ) -> None:
        r"""Initializes the Ising PSAM.

        Args:
            coupling: The initial coupling :math:`J`, one value per group.
            train_coupling: Whether to train :math:`J`.
            coupling_schedule: The values of :math:`J` stepped through by
                `update_binding_optim`, in order.
            coupling_with_monomer: Whether to unfreeze :math:`J` in the same
                step as the monomer betas, so that it is free from the start,
                rather than in a step of its own afterwards.
            coupling_groups: The group index of each nucleotide interface, so
                that one parameter can be shared by several; inferred from the
                length of `coupling` if None.
            method: Whether to expand the filter over microstates or
                positions.
        """
        super().__init__(*args, **kwargs)
        if self.pairwise_distance != 0:
            raise ValueError("IsingPSAM does not support pairwise features")
        if method not in ("enumerate", "transfer"):
            raise ValueError(f"unknown method {method!r}")
        # The buffers below are sized by kernel_size, so a heuristic that
        # resized the footprint would silently invalidate them
        for flag in (
            "shift_footprint",
            "shift_footprint_heuristic",
            "increment_footprint",
            "increment_flank_with_footprint",
        ):
            if getattr(self, flag):
                raise ValueError(f"IsingPSAM does not support {flag}")
        self.method = method
        n_interfaces = self.kernel_size - 1

        self.sigma_on: Tensor
        self.sigma_pair: Tensor
        self.position_mask: Tensor
        self.coupling_groups: Tensor
        if method == "enumerate":
            sigma = torch.tensor(
                list(itertools.product([1.0, -1.0], repeat=self.kernel_size)),
                dtype=torch.float32,
            )
            self.register_buffer("sigma_on", (sigma + 1) / 2)
            self.register_buffer("sigma_pair", sigma[:, :-1] * sigma[:, 1:])
        else:
            self.register_buffer("position_mask", torch.eye(self.kernel_size))

        if coupling_groups is None:
            n_given = torch.as_tensor(coupling).numel()
            if n_given == 1:
                coupling_groups = torch.zeros(n_interfaces, dtype=torch.long)
            elif n_given == n_interfaces:
                coupling_groups = torch.arange(n_interfaces)
            else:
                raise ValueError(
                    f"coupling has {n_given} elements, expected 1 or"
                    f" {n_interfaces}, or pass coupling_groups"
                )
        coupling_groups = torch.as_tensor(coupling_groups, dtype=torch.long)
        if coupling_groups.shape != (n_interfaces,):
            raise ValueError(
                f"coupling_groups must have shape ({n_interfaces},)"
            )
        self.register_buffer("coupling_groups", coupling_groups)

        init = torch.as_tensor(coupling, dtype=torch.float32).flatten()
        n_groups = int(coupling_groups.max()) + 1
        if init.numel() == 1:
            init = init.expand(n_groups).clone()
        elif init.numel() == n_interfaces:
            init = torch.stack(
                [init[coupling_groups == g].mean() for g in range(n_groups)]
            )
        self.coupling = torch.nn.Parameter(init, requires_grad=False)
        self.train_coupling = train_coupling
        self.coupling_schedule = tuple(coupling_schedule)
        self.coupling_with_monomer = coupling_with_monomer

    @property
    def n_expanded(self) -> int:
        """The number of filter channels per output channel."""
        # One per microstate, or one per position for the transfer matrices
        kernel_size = int(self.kernel_size)
        if self.method == "enumerate":
            return int(2**kernel_size)
        return kernel_size

    def interface_coupling(self) -> Tensor:
        r"""The coupling :math:`J_i` of each nucleotide interface."""
        return self.coupling[self.coupling_groups]

    def is_reverse(self, channel: int) -> bool:
        """Whether an output channel scores the reverse strand."""
        # get_filter orders channels Afor Bfor ... Brev Arev
        return channel >= self.out_channels // self.n_strands

    def coupling_bias(self) -> Tensor:
        r"""The coupling energy of each microstate, of shape
        :math:`(\text{out_channels},2^{\text{kernel_size}})`."""
        # J is reversed for reverse-strand channels, whose position axis is
        # flipped by get_filter; one coupling is shared by all motifs
        coupling = self.interface_coupling()
        return torch.stack(
            [
                (
                    (self.sigma_pair - 1.0)
                    * (coupling.flip(0) if self.is_reverse(i) else coupling)
                ).sum(-1)
                for i in range(self.out_channels)
            ]
        )

    def log_transfer(self, reverse: bool = False) -> Tensor:
        r"""The log transfer matrices, of shape
        :math:`(\text{kernel_size} - 1, 2, 2)`.

        In the states :math:`(+1, -1)`, :math:`\mathcal{T}_i` is 1 on the
        diagonal and :math:`e^{-2 J_i}` off it. Set `reverse` for
        reverse-strand channels.
        """
        coupling = self.interface_coupling()
        if reverse:
            coupling = coupling.flip(0)
        zero, off = torch.zeros_like(coupling), -2.0 * coupling
        return torch.stack(
            [
                torch.stack([zero, off], dim=-1),
                torch.stack([off, zero], dim=-1),
            ],
            dim=-2,
        )

    @override
    def get_filter(self, dist: int) -> Tensor:
        r"""PSAM filter expanded over microstates or positions, of shape
        :math:`(\text{out_channels}\times\text{n_expanded},
        \text{in_channels},\text{kernel_size})`."""
        base = super().get_filter(dist)
        mask = (
            self.sigma_on if self.method == "enumerate" else self.position_mask
        )
        expanded = mask[None, :, None, :] * base[:, None, :, :]
        return expanded.reshape(-1, base.shape[-2], base.shape[-1])

    @override
    def get_logo_filter(self, dist: int = 0) -> Tensor:
        return super().get_filter(dist)

    @override
    def get_bias(self) -> Tensor:
        # One copy per filter channel, since Conv1d scores them all
        return super().get_bias().repeat_interleave(self.n_expanded, dim=0)

    def set_coupling(self, value: float) -> None:
        r"""Sets :math:`J` for every group."""
        with torch.no_grad():
            self.coupling.fill_(value)

    @override
    def unfreeze(self, parameter: unfreezable = "all") -> None:
        if self.train_coupling and parameter in ("coupling", "all"):
            self.coupling.requires_grad_()
        if parameter != "coupling":
            super().unfreeze(parameter)

    @override
    def update_binding_optim(
        self, binding_optim: BindingOptim
    ) -> BindingOptim:
        # Steps J through coupling_schedule, then unfreezes it. The schedule
        # must increase from J ~ 0, where the chain is decoupled and each
        # beta_i has its own gradient; at large J the window only flips as a
        # whole and the gradient vanishes
        binding_optim = super().update_binding_optim(binding_optim)

        insertion_idx = len(binding_optim.steps)
        monomer_step: Step | None = None
        for step_idx, step in enumerate(binding_optim.steps):
            for call in step.calls:
                if (
                    call.cmpt is self
                    and call.fun == "unfreeze"
                    and call.kwargs.get("parameter") == "monomer"
                ):
                    insertion_idx = max(step_idx + 1, insertion_idx)
                    monomer_step = step

        # The activity holds E_0, and is otherwise frozen until the last step
        if monomer_step is not None:
            for cmpt in {c for anc in binding_optim.ancestry for c in anc}:
                if isinstance(cmpt, Contribution):
                    monomer_step.calls.append(
                        Call(cmpt, "unfreeze", {"parameter": "activity"})
                    )

        for value in self.coupling_schedule:
            if value == 0.0:
                continue
            binding_optim.steps.insert(
                insertion_idx,
                Step([Call(self, "set_coupling", {"value": value})]),
            )
            insertion_idx += 1

        if self.train_coupling:
            call = Call(self, "unfreeze", {"parameter": "coupling"})
            if self.coupling_with_monomer and monomer_step is not None:
                monomer_step.calls.append(call)
            else:
                binding_optim.steps.insert(insertion_idx, Step([call]))

        binding_optim.merge_binding_optim()
        return binding_optim


class IsingConv1d(Conv1d):
    r"""1d convolution reducing the expanded channels of an `IsingPSAM`.

    `Conv1d` scores every filter channel of every window; this sums their
    Boltzmann weights to give the :math:`-\log K^{rel}_{\text{D}}` of each
    window, over microstates for `method="enumerate"` and over transfer
    matrices for `method="transfer"`.

    The :math:`\sigma = -1` state contributes a weight of 1 to every window, so
    a `NonSpecific` mode would double-count the alternative binding mode.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.layer_spec, IsingPSAM):
            raise TypeError("IsingConv1d requires an IsingPSAM layer_spec")
        # out_channel_indexing selects the PSAM's output channels; expand it to
        # the matching filter channels so inherited scoring indexes correctly
        self._base_indexing: list[int] | None = self._out_channel_indexing
        if self._base_indexing is not None:
            n_expanded = self.layer_spec.n_expanded
            self._out_channel_indexing = [
                channel * n_expanded + i
                for channel in self._base_indexing
                for i in range(n_expanded)
            ]

    @override
    def _get_log_posbias_indexed(self, seqs: Tensor) -> Tensor | None:
        r"""Always None, so that the inherited scoring adds no bias.

        :math:`\omega(x)` is a per-window offset on the window's log score, so
        it is added by `forward` after the reduction. For `"enumerate"` the two
        are equivalent, but the transfer recursion is not linear in the
        per-position scores.
        """
        return None

    @property
    def base_channels(self) -> list[int]:
        """The PSAM output channel scored by each output channel."""
        if self._base_indexing is not None:
            return list(self._base_indexing)
        return list(range(self.layer_spec.out_channels))

    @override
    @property
    def out_channels(self) -> int:
        return len(self.base_channels)

    @staticmethod
    def _transfer(betas: Tensor, log_transfer: Tensor) -> Tensor:
        r"""Sums over :math:`\sigma` by eliminating one position at a time.

        Args:
            betas: The :math:`\beta_{\phi_i}` of each position, of shape
                :math:`(\ldots,\text{kernel_size})`.
            log_transfer: The log transfer matrices from `log_transfer`.
        """
        zero = torch.zeros_like(betas[..., 0])
        state = torch.stack([betas[..., 0], zero], dim=-1)
        for i in range(betas.shape[-1] - 1):
            state = torch.logsumexp(
                state.unsqueeze(-1) + log_transfer[i], dim=-2
            )
            state = state + torch.stack([betas[..., i + 1], zero], dim=-1)
        return torch.logsumexp(state, dim=-1)

    @override
    def forward(self, seqs: Tensor) -> Tensor:
        spec = self.layer_spec
        if not isinstance(spec, IsingPSAM):
            raise TypeError("IsingConv1d requires an IsingPSAM layer_spec")
        channels = self.base_channels
        out = super().forward(seqs)
        out = out.reshape(
            out.shape[0], len(channels), spec.n_expanded, out.shape[-1]
        )

        if spec.method == "enumerate":
            out = out + spec.coupling_bias()[channels][None, :, :, None]
            score = out.logsumexp(dim=2)
        else:
            betas = out.movedim(2, -1)
            score = torch.stack(
                [
                    self._transfer(
                        betas[:, i],
                        spec.log_transfer(spec.is_reverse(channel)),
                    )
                    for i, channel in enumerate(channels)
                ],
                dim=1,
            )
            # A window over padding makes every filter channel -inf, whereas a
            # -inf beta only makes its own channel -inf; the recursion handles
            # the latter but would give the former a weight of 1
            score = score.masked_fill(betas.isneginf().all(-1), float("-inf"))

        posbias = Conv1d._get_log_posbias_indexed(self, seqs)
        if posbias is not None:
            score = score + posbias
        return score
