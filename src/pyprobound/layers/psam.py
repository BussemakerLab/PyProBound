"""Position-specific affinity matrix (PSAM).

Members are explicitly re-exported in pyprobound.layers.
"""

import math
from collections.abc import Sequence, Set
from typing import Any, Literal, TypeVar, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

from .. import __precision__
from ..alphabets import Alphabet
from ..base import BindingOptim, Call, Step
from ..containers import TParameterDict
from .layer import LayerSpec

T = TypeVar("T", int, Tensor)
Conv1d = Any


class PSAM(LayerSpec):
    r"""PSAM parameters with symmetry encoding and pairwise features.

    The PSAM is a sliding filter where the weight :math:`\beta_\phi` of feature
    :math:`\phi` is defined as :math:`-\Delta\Delta G_\phi/RT`.
    A feature :math:`\phi` can be the presence of a letter at a position,
    or a pair of two letters at two different positions.

    Attributes:
        betas (TParameterDict): The sequence-specific parameters.
    """

    unfreezable = Literal[LayerSpec.unfreezable, "monomer", "pairwise"]

    def __init__(
        self,
        kernel_size: int,
        alphabet: Alphabet | None = None,
        out_channels: int | None = None,
        in_channels: int | None = None,
        pairwise_distance: int = 0,
        symmetry: Sequence[int] | None = None,
        seed: Sequence[Sequence[str]] | None = None,
        seed_scale: int = 1,
        score_reverse: bool | None = None,
        shift_footprint: bool = False,
        shift_footprint_heuristic: bool = False,
        increment_footprint: bool = False,
        increment_flank: bool = False,
        increment_flank_with_footprint: bool = False,
        information_threshold: float = 0.1,
        max_kernel_size: int | None = None,
        frozen_parameters: Set[str] = frozenset(),
        normalize: bool = False,
        train_betas: bool = True,
        name: str = "",
    ) -> None:
        """Initializes the PSAM.

        Args:
            kernel_size: The size of the convolving PSAM kernel.
            alphabet: The alphabet used to encode sequences into tensors. Must
                be provided if input sequences are not embedded, or if
                `out_channels` or `in_channels` are not specified.
            out_channels: The number of output channels, inferred from
                `score_reverse` if not specified. If `score_reverse`, must be
                even, with half the channels representing the complement.
            in_channels: The number of input channels, inferred from `alphabet`
                if not specified.
            pairwise_distance: The distance between two positions on the PSAM
                for which pairwise letter features will be scored.
            symmetry: An encoding of reverse-complement and translational
                symmetries. All positions with the same integer will share
                parameters, while two positions with opposite signs will
                be complementary. For example, `[1,2,3,-3,-2,-1]` encodes a
                reverse-complement symmetric PSAM.
            seed: A seed string for each non-reverse complement output channel.
                Any character in the alphabet's encoding may be used; for
                example, `["TATAWAW"]` might be a seed for the TATA box.
            seed_scale: A scaling factor for the strength of the seed.
            score_reverse: Whether to score the reverse strand, inferred
                from `alphabet` if not specified, defaults to False if neither.
            shift_footprint: Whether to add a greedy exploration of shifts of
                positions on the PSAM to the sequential optimization procedure.
                For example, given a symmetry vector `[2,3,4,5]`, will attempt
                `[3,4,5,6]` and `[1,2,3,4]` to try to escape local optima.
            shift_footprint_heuristic: Like `shift_footprint`, but in one step
                by calculating the center of mass of the information content.
            increment_footprint: Whether to add a greedy exploration of the
                kernel size to the sequential optimization procedure. For
                example, given a symmetry vector `[2,3,4,5]`, will attempt
                `[1,2,3,4,5,6]`.
            increment_flank_with_footprint: Whether to increment the flank
                length with the footprint to keep the output length constant.
            information_threshold: The minimum information in the first two and
                last two positions on the PSAM for incrementing the footprint.
            max_kernel_size: The maximum kernel size allowed.
            frozen_parameters: The name of the parameters in `betas` which will
                never be trained.
            normalize: Whether to mean-center the PSAM. A separate `bias`
                parameter is trained for the convolution in its place.
            train_betas: Whether to train any PSAM parameters, used to restrict
                gradient calculation to sequence-independent parameters only.
            name: A string used to describe the PSAM.
        """
        # Fill in defaults
        if score_reverse is None:
            if alphabet is None:
                score_reverse = False
            else:
                score_reverse = alphabet.complement
        if in_channels is None:
            if alphabet is None:
                raise ValueError(
                    "At least one of alphabet or in_channels must be specified"
                )
            in_channels = len(alphabet.alphabet)
        n_strands = 2 if score_reverse else 1
        if out_channels is None:
            out_channels = n_strands
        if symmetry is None:
            symmetry = list(range(1, kernel_size + 1))

        # Call super
        super().__init__(
            out_channels=out_channels, in_channels=in_channels, name=name
        )
        self._layers: set[Conv1d]

        # Check input
        if out_channels % n_strands != 0:
            raise ValueError(
                f"out_channels ({out_channels}) not divisible by"
                f" the number of scored strands ({n_strands})"
            )
        if any(i == 0 for i in symmetry):
            raise ValueError("symmetry cannot include zeros")
        if len(symmetry) != kernel_size:
            raise ValueError("symmetry must be length kernel_size")
        if not (kernel_size > 0 and in_channels > 0 and out_channels > 0):
            raise ValueError(
                "kernel_size, in_channels, and out_channels must be positive"
            )
        if not 0 <= pairwise_distance < kernel_size:
            raise ValueError("pairwise_distance must be in [0, kernel_size)")

        # Store instance attributes
        self.alphabet = alphabet
        self.frozen_parameters = frozen_parameters
        self._n_strands = cast(Literal[1, 2], n_strands)
        self._pairwise_distance = pairwise_distance
        self._score_reverse = score_reverse
        self.normalize = normalize
        self.shift_footprint = shift_footprint
        self.shift_footprint_heuristic = shift_footprint_heuristic
        self.increment_footprint = increment_footprint
        self.increment_flank = increment_flank
        self.increment_flank_with_footprint = increment_flank_with_footprint
        self.information_threshold = information_threshold
        self.max_kernel_size = max_kernel_size
        self.train_betas = train_betas
        self.symmetry: Tensor
        self.register_buffer("symmetry", torch.tensor(symmetry))

        # Create and initialize matrix parameters
        self.bias = torch.nn.Parameter(
            torch.zeros(
                (self.out_channels // self.n_strands, 1), dtype=__precision__
            )
        )
        self.betas: TParameterDict = TParameterDict()
        self.update_params(pairwise_grad=True)
        if seed is not None:
            self._seed(seed, seed_scale=seed_scale)

    @property
    def kernel_size(self) -> int:
        """The size of the convolving PSAM kernel."""
        return len(self.symmetry)

    @property
    def n_strands(self) -> Literal[1, 2]:
        """The number of strands scored by the PSAM."""
        return self._n_strands

    @property
    def pairwise_distance(self) -> int:
        """The distance between two positions with pairwise letter features."""
        return self._pairwise_distance

    @property
    def score_reverse(self) -> bool:
        """Whether to score the reverse strand."""
        return self._score_reverse

    @override
    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        del mode
        return length - self.kernel_size + 1

    @override
    def in_len(self, length: T, mode: Literal["min", "max"] = "max") -> T:
        del mode
        return self.kernel_size + length - 1

    @staticmethod
    def _get_key(elements: tuple[Tensor, Tensor], index: int) -> str:
        """Key for a parameter in `betas`.

        Args:
            elements: The two positions on the symmetry vector. For monomer
                features, both elements should be identical.
            index: The index on the parameter vector.

        Returns:
            The key for the parameter in the `betas` TParameterDict.
        """
        iter_elements = (
            elements[:1] if elements[-1] == elements[0] else elements
        )
        return "-".join(
            [
                str(i)
                for i in sorted([abs(i.cpu().item()) for i in iter_elements])
            ]
            + [str(index)]
        )

    @staticmethod
    def frozen_positions(
        positions: list[int],
        symmetry: list[int],
        out_channels: int,
        in_channels: int,
    ) -> set[str]:
        """The keys of `betas` parameters to be frozen.

        Args:
            positions: The positions on the symmetry that will be frozen.
            symmetry: The vector encoding reverse-complement and translational
                symmetries.
            out_channels: The number of output channels.
            in_channels: The number of input channels.

        Returns:
            The keys of `betas` parameters to be passed to `__init__`.
        """
        out: set[str] = set()
        for pos in positions:
            for pos_alt in symmetry + list(
                range(max(symmetry), max(symmetry) + len(symmetry))
            ):
                num_index: int = out_channels * in_channels
                if (pos_alt - pos) != 0:
                    num_index *= in_channels
                for index in range(num_index):
                    out.add(
                        PSAM._get_key(
                            (torch.tensor(pos), torch.tensor(pos_alt)), index
                        )
                    )
        return out

    def _seed(
        self, seed: Sequence[Sequence[str]], seed_scale: int = 1
    ) -> None:
        """Seed monomer feature vectors with given values.

        Args:
            seed: A seed string for each non-reverse complement output channel.
                Any character in the alphabet's encoding may be used; for
                example, `["TATAWAW"]` might be a seed for the TATA box.
            seed_scale: A scaling factor for the strength of the seed.
        """

        if self.alphabet is None:
            raise ValueError("alphabet must be specified if seed is specified")
        if len(seed) != self.out_channels // self.n_strands:
            raise ValueError(
                "Need as many seeds as there are unique out_channels"
            )
        if any(len(s) != self.kernel_size for s in seed):
            raise ValueError("Each seed must be of length kernel_size")

        # Stdev of non-seeded ~ Uniform[± 1/sqrt(in_channels * kernel_size)]
        stdev = math.sqrt(1 / (3 * self.in_channels * self.kernel_size))

        for out_channel, channel_seed in enumerate(seed):
            for position, seed_code in zip(self.symmetry, channel_seed):
                encoding = self.alphabet.get_encoding[seed_code]
                if position < 0:
                    encoding = tuple(
                        len(self.alphabet.alphabet) - i - 1 for i in encoding
                    )

                if len(encoding) == 0:  # Don't seed
                    continue
                if len(encoding) == self.in_channels:  # Zero out
                    seeded_fill = 0.0
                    unseeded_fill = 0.0
                else:
                    multiplier = math.sqrt(
                        (self.in_channels - len(encoding)) / len(encoding)
                    )
                    seeded_fill = seed_scale * stdev * multiplier
                    unseeded_fill = -seed_scale * stdev / multiplier
                    # Ensure Var[seeded] = Var[non-seeded]
                    # P(seeded=seeded_fill) = len(encoding) / in_channels
                    # P(seeded-unseeded_fill) = 1 - P(seeded=seeded_fill)

                for index in range(len(self.alphabet.alphabet)):
                    if position < 0:
                        index = self.in_channels - index - 1
                    j = out_channel * self.in_channels + index
                    parameter = self.betas[
                        self._get_key((position, position), j)
                    ]
                    val = seeded_fill if index in encoding else unseeded_fill
                    torch.nn.init.constant_(parameter, val)

    @override
    def unfreeze(self, parameter: unfreezable = "all") -> None:
        if self.train_betas:
            if parameter in ("monomer", "pairwise", "all"):
                if self.normalize:
                    self.bias.requires_grad_()
                for key, param in self.betas.items():
                    pairwise = len(key.split("-")) == 3
                    if (
                        parameter in ("monomer", "all")
                        and not pairwise
                        and key not in self.frozen_parameters
                    ):
                        param.requires_grad_()
                    elif (
                        parameter in ("pairwise", "all")
                        and pairwise
                        and key not in self.frozen_parameters
                    ):
                        param.requires_grad_()
        # pylint: disable-next=consider-using-in
        if parameter != "monomer" and parameter != "pairwise":
            super().unfreeze(parameter)

    @override
    def update_binding_optim(
        self, binding_optim: BindingOptim
    ) -> BindingOptim:
        if self.kernel_size > 0:
            binding_optim.steps.append(
                Step([Call(self, "unfreeze", {"parameter": "monomer"})])
            )

        # Greedy search
        if self.increment_flank:
            calls: list[Call] = []
            for conv1d in self._layers:
                # pylint: disable-next=protected-access
                for bmd in set(bmd[0] for bmd in conv1d._modes):
                    calls.append(
                        Call(
                            bmd,
                            "update_read_length",
                            {"left_shift": 1, "right_shift": 1},
                        )
                    )
            binding_optim.steps.append(Step(calls, greedy=True))

        if self.shift_footprint_heuristic:
            binding_optim.steps.append(
                Step([Call(self, "shift_heuristic", {})], greedy=True)
            )

        if self.shift_footprint:
            binding_optim.steps.extend(
                [
                    Step(
                        [
                            Call(
                                self,
                                "update_footprint",
                                {"left_shift": 1, "right_shift": -1},
                            )
                        ],
                        greedy=True,
                    ),
                    Step(
                        [
                            Call(
                                self,
                                "update_footprint",
                                {"left_shift": -1, "right_shift": 1},
                            )
                        ],
                        greedy=True,
                    ),
                ]
            )

        if self.increment_footprint:
            binding_optim.steps.append(
                Step(
                    [
                        Call(
                            self,
                            "update_footprint",
                            {
                                "left_shift": 1,
                                "right_shift": 1,
                                "check_threshold": True,
                            },
                        )
                    ],
                    greedy=True,
                )
            )
            if self.increment_flank_with_footprint:
                for conv1d in self._layers:
                    # pylint: disable-next=protected-access
                    for bmd in set(bmd[0] for bmd in conv1d._modes):
                        binding_optim.steps[-1].calls.append(
                            Call(
                                bmd,
                                "update_read_length",
                                {"left_shift": 1, "right_shift": 1},
                            )
                        )

        # Interaction
        if self.pairwise_distance > 0:
            binding_optim.steps.append(
                Step([Call(self, "unfreeze", {"parameter": "pairwise"})])
            )

        binding_optim.merge_binding_optim()
        return binding_optim

    def update_params(
        self, pairwise_grad: bool = False
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Update the `betas` according to the symmetry string.

        Args:
            pairwise_grad: Whether enable gradient calculation on new pairwise
                parameters. Monomer parameters always have it enabled.

        Returns:
            A tuple of the added and removed parameters, respectively.
        """
        added_params = []
        removed_keys = set(self.betas.keys())
        out_channels = self.out_channels // self.n_strands

        for dist in range(self.pairwise_distance + 1):
            shape: tuple[int, ...]
            if dist == 0:
                shape = (out_channels, self.in_channels)
            else:
                shape = (out_channels, self.in_channels, self.in_channels)

            for i in range(self.kernel_size - dist):
                for j in range(math.prod(shape)):
                    key = self._get_key(
                        (self.symmetry[i], self.symmetry[i + dist]), j
                    )
                    if key in self.betas:
                        removed_keys.discard(key)
                    else:
                        param = torch.nn.Parameter(
                            torch.tensor(
                                0.0,
                                device=self.symmetry.device,
                                dtype=__precision__,
                            ),
                            requires_grad=(
                                (pairwise_grad or dist == 0)
                                and key not in self.frozen_parameters
                            ),
                        )
                        if dist == 0:
                            bound = math.sqrt(
                                1 / (self.in_channels * self.kernel_size)
                            )
                            torch.nn.init.uniform_(param, -bound, bound)
                        self.betas[key] = param
                        added_params.append(param)

        removed_params = tuple(self.betas.pop(key) for key in removed_keys)

        return tuple(added_params), tuple(removed_params)

    def update_footprint(
        self,
        left_shift: int = 0,
        right_shift: int = 0,
        check_threshold: bool = False,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Extend or shrink the symmetry in either direction.

        Args:
            left_shift: The number of positions to add to the left of the PSAM.
            right_shift: The number of positions to add to the right of the
                PSAM.
            check_threshold: Whether to ensure that the information content of
                the first and last two positions are above
                `information_threshold`.

        Returns:
            A tuple of the added and removed parameters, respectively.
        """
        if left_shift == right_shift == 0:
            return tuple(), tuple()
        if left_shift + right_shift <= -(
            self.kernel_size - self.pairwise_distance
        ):
            raise ValueError(
                "Cannot shrink footprint beyond the last pairwise distance"
            )
        if self.max_kernel_size is not None and (
            left_shift + right_shift + self.kernel_size > self.max_kernel_size
        ):
            raise ValueError(
                f"left_shift={left_shift}, right_shift={right_shift} results "
                f"in a size of {self.kernel_size + left_shift + right_shift}"
                f" but max_kernel_size is {self.max_kernel_size}"
            )

        # Check information content if check_threshold
        info_left = [self.get_information(0)]
        info_right = [self.get_information(-1)]
        if self.kernel_size > 1:
            info_left.append(self.get_information(1))
            info_right.append(self.get_information(-2))
        if check_threshold:
            if left_shift > 0 and min(info_left) < self.information_threshold:
                left_shift = 0
            if left_shift < 0 and max(info_left) > self.information_threshold:
                left_shift = 0
            if (
                right_shift > 0
                and min(info_right) < self.information_threshold
            ):
                right_shift = 0
            if (
                right_shift < 0
                and max(info_right) > self.information_threshold
            ):
                right_shift = 0
            if left_shift == 0 and right_shift == 0:
                raise ValueError(
                    f"Cannot shift footprint ({left_shift, right_shift})"
                    f" since information content is {info_left} on the left"
                    f" and {info_right} on the right"
                    f"; {self.information_threshold} needed for shift"
                )

        # Update biases
        for conv1d in self._layers:
            # pylint: disable=protected-access
            conv1d._update_biases(
                binding_mode_left=left_shift,
                binding_mode_right=right_shift,
                window_top=-left_shift,
                window_bottom=-right_shift,
            )

        # Update symmetry string
        next_val = (self.symmetry.max() + 1).unsqueeze(0)
        self.symmetry = self.symmetry[
            max(0, -left_shift) : self.kernel_size - max(0, -right_shift)
        ]
        for _ in range(left_shift):  # prepend
            self.symmetry = torch.cat((next_val, self.symmetry), dim=0)
            next_val += 1
        for _ in range(right_shift):  # append
            self.symmetry = torch.cat((self.symmetry, next_val), dim=0)
            next_val += 1

        # Propagate updates
        for conv1d in self._layers:
            # pylint: disable=protected-access
            for bmd, layer_idx in conv1d._modes:
                bmd._update_propagation(
                    layer_idx + 1,
                    left_shift=-left_shift,
                    right_shift=-right_shift,
                )
            # pylint: enable=protected-access

        # Update parameters
        return self.update_params(pairwise_grad=False)

    def shift_heuristic(self) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Shift the footprint to center the information COM.

        Returns:
            A tuple of the added and removed parameters, respectively.
        """
        informations = [
            self.get_information(idx) for idx, _ in enumerate(self.symmetry)
        ]
        center_position = sum(
            idx * info for idx, info in enumerate(informations)
        ) / sum(informations)
        right_shift = round(center_position - ((self.kernel_size - 1) / 2))
        return self.update_footprint(-right_shift, right_shift)

    def get_information(self, position: int) -> float:
        """Get the information content of the position on the PSAM."""
        with torch.inference_mode():
            psam = self.get_filter(0)
            psam = psam[: len(psam) // self.n_strands, :, position].flatten()
        probability = F.softmax(psam, dim=0)
        information = torch.log2(
            torch.tensor(probability.numel(), device=probability.device)
        )
        information += torch.sum(probability * torch.log2(probability))
        return information.item()

    def get_dirichlet(self) -> Tensor:
        """Calculates the Dirichlet-inspired regularization value (see pub)."""
        # Combine parameters into parameter classes (all betas at a position)
        param_classes: dict[str, set[Tensor]] = {}
        for name, param in self.betas.items():
            key = name.rsplit("-", 1)[0]  # Drop index from key
            if key in param_classes:
                param_classes[key].add(param)
            else:
                param_classes[key] = {param}

        regularization = torch.tensor(0.0, device=self.symmetry.device)
        for param_class in param_classes.values():
            vector = torch.stack(tuple(param_class))
            regularization += torch.sum(vector) - (
                vector.numel() * torch.logsumexp(vector, 0)
            )
        return regularization

    def get_bias(self) -> Tensor:
        """A bias parameter passed to Conv1d."""
        if self.score_reverse:
            return torch.cat((self.bias, self.bias.flip(0)))
        return self.bias

    def get_filter(self, dist: int) -> Tensor:
        """PSAM filter for a given pairwise distance."""
        if not 0 <= dist <= self.pairwise_distance:
            raise ValueError("dist must be in [0, self.pairwise_distance]")

        params = []
        for i in range(self.kernel_size - dist):
            elements = (self.symmetry[i], self.symmetry[i + dist])

            shape: tuple[int, ...]
            if dist == 0:
                shape = (self.out_channels // self.n_strands, self.in_channels)
            else:
                shape = (
                    self.out_channels // self.n_strands,
                    self.in_channels,
                    self.in_channels,
                )

            param = torch.stack(
                [
                    self.betas[self._get_key(elements, j)]
                    for j in range(math.prod(shape))
                ]
            ).view(shape)
            if dist > 0 and self.symmetry[i] == self.symmetry[i + dist]:
                # Autosymmetric pairwise parameter
                param = param.triu() + param.triu(1).T
            elements_sort = [i.cpu().abs().item() for i in elements]
            if elements_sort != sorted(elements_sort):
                param.transpose_(-1, -2)
            if self.symmetry[i] < 0:
                param = param.flip(-1)
            # pylint: disable-next=chained-comparison
            if dist > 0 and self.symmetry[i + dist] < 0:
                param = param.flip(-2)
            if self.normalize:
                param = param - param.mean()
            params.append(param)

        matrix = torch.stack(params, dim=-1)
        if self.score_reverse:
            # First dimension is Afor Bfor Brev Arev
            if dist == 0:
                matrix_reverse = matrix.flip(dims=[0, 1, 2])
            else:
                matrix_reverse = matrix.flip(dims=[0, 1, 2, 3]).transpose(1, 2)
            matrix = torch.cat((matrix, matrix_reverse))
        return matrix

    def fix_gauge(self) -> None:
        """Removes invariances between monomer and pairwise parameters."""
        bias_shift = 0.0
        for dist in range(1, self.pairwise_distance + 1):
            for i in range(self.kernel_size - dist):
                feat1, feat2 = self.symmetry[i], self.symmetry[i + dist]

                pairwise_params = [
                    self.betas[self._get_key((feat1, feat2), j)]
                    for j in range(
                        (self.out_channels // self.n_strands)
                        * self.in_channels**2
                    )
                ]
                pairwise_param = torch.stack(
                    cast(list[Tensor], pairwise_params)
                ).view(
                    (
                        self.out_channels // self.n_strands,
                        self.in_channels,
                        self.in_channels,
                    )
                )

                monomer1_params = [
                    self.betas[self._get_key((feat1, feat1), j)]
                    for j in range(
                        (self.out_channels // self.n_strands)
                        * self.in_channels
                    )
                ]
                monomer1_param = torch.stack(
                    cast(list[Tensor], monomer1_params)
                ).view((self.out_channels // self.n_strands, self.in_channels))
                monomer2_params = [
                    self.betas[self._get_key((feat2, feat2), j)]
                    for j in range(
                        (self.out_channels // self.n_strands)
                        * self.in_channels
                    )
                ]
                monomer2_param = torch.stack(
                    cast(list[Tensor], monomer2_params)
                ).view((self.out_channels // self.n_strands, self.in_channels))

                shift1 = pairwise_param.mean(dim=2, keepdim=True)
                pairwise_param -= shift1
                shift2 = pairwise_param.mean(dim=1, keepdim=True)
                pairwise_param -= shift2
                monomer1_param += shift1.reshape(shift1.shape[0], -1)
                monomer2_param += shift2.reshape(shift2.shape[0], -1)
                if self.normalize:
                    bias_shift += (
                        monomer1_param.mean(-1) + monomer2_param.mean(-1)
                    ).item()
                    monomer1_param -= monomer1_param.mean(-1)
                    monomer2_param -= monomer2_param.mean(-1)

                for param, val in zip(
                    pairwise_params + monomer1_params + monomer2_params,
                    torch.cat(
                        [
                            i.flatten()
                            for i in (
                                pairwise_param,
                                monomer1_param,
                                monomer2_param,
                            )
                        ]
                    ),
                ):
                    torch.nn.init.constant_(param, val)

        self.bias.data = self.bias + bias_shift
