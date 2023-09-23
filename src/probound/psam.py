"""ProBound convolutional models"""
import math
from collections.abc import Sequence, Set
from typing import Any, Literal, TypeVar, cast, overload

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

from . import __precision__
from .alphabet import Alphabet
from .base import BindingOptim, Call, Step
from .containers import Buffer, TParameterDict
from .layers import LayerSpec

T = TypeVar("T", int, Tensor)
Conv1d = Any


class PSAM(LayerSpec):
    """PSAM parameters with symmetry encoding and interaction features"""

    _layers: set[Conv1d]

    _unfreezable = Literal[LayerSpec._unfreezable, "mono", "interaction"]

    def __init__(
        self,
        kernel_size: int,
        alphabet: Alphabet | None = None,
        out_channels: int | None = None,
        in_channels: int | None = None,
        interaction_distance: int = 0,
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
        # fill in defaults
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

        # call super
        super().__init__(
            out_channels=out_channels, in_channels=in_channels, name=name
        )

        # check input
        if out_channels % n_strands != 0:
            raise ValueError(
                f"out_channels ({out_channels}) not divisible by"
                f" the number of scored strands ({n_strands})"
            )
        if any(i == 0 for i in symmetry):
            raise ValueError("symmetry cannot include zeros")
        if len(symmetry) != kernel_size:
            raise ValueError("symmetry must be length kernel_size")
        if not (kernel_size >= 0 and in_channels > 0 and out_channels > 0):
            raise ValueError(
                "kernel_size must be non-negative,"
                " and channels must be positive"
            )
        if kernel_size == 0 and (
            interaction_distance != 0 or out_channels != 1
        ):
            raise ValueError(
                "If kernel_size is 0 interaction_distance must be"
                " 0, out_channels=bias_bin=1, train_theta=one_hot=False"
            )
        if not (
            0 <= interaction_distance < (kernel_size if kernel_size > 0 else 1)
        ):
            raise ValueError(
                "interaction_distance must be in [0, kernel_size)"
            )

        # store instance attributes
        self.alphabet = alphabet
        self.frozen_parameters = frozen_parameters
        self.symmetry: Tensor = Buffer(torch.tensor(symmetry))
        self._in_channels: Tensor = Buffer(torch.tensor(in_channels))
        self._out_channels: Tensor = Buffer(torch.tensor(out_channels))
        self._n_strands: Tensor = Buffer(torch.tensor(n_strands))
        self._interaction_distance: Tensor = Buffer(
            torch.tensor(interaction_distance)
        )
        self._score_reverse: Tensor = Buffer(torch.tensor(score_reverse))
        self._normalize: Tensor = Buffer(torch.tensor(normalize))
        self.shift_footprint = shift_footprint
        self.shift_footprint_heuristic = shift_footprint_heuristic
        self.increment_footprint = increment_footprint
        self.increment_flank = increment_flank
        self.increment_flank_with_footprint = increment_flank_with_footprint
        self.information_threshold = information_threshold
        self.max_kernel_size = max_kernel_size
        self.train_betas = train_betas

        # create and initialize matrix parameters
        self.bias = torch.nn.Parameter(
            torch.zeros(
                (self.out_channels // self.n_strands, 1), dtype=__precision__
            )
        )
        self.betas: TParameterDict = TParameterDict()
        self.update_params(di_grad=True)
        if seed is not None:
            self._seed(seed, seed_scale=seed_scale)

    @property
    def n_strands(self) -> int:
        return cast(int, self._n_strands.item())

    @property
    def interaction_distance(self) -> int:
        return cast(int, self._interaction_distance.item())

    @property
    def score_reverse(self) -> bool:
        return cast(bool, self._score_reverse.item())

    @property
    def normalize(self) -> bool:
        return cast(bool, self._normalize.item())

    @override
    def unfreeze(self, parameter: _unfreezable = "all") -> None:
        """Unfreeze the desired parameter"""
        if self.train_betas:
            if parameter in ("mono", "interaction", "all"):
                if self.normalize:
                    self.bias.requires_grad_()
                for key, param in self.betas.items():
                    interaction = len(key.split("-")) == 3
                    if (
                        parameter in ("mono", "all")
                        and not interaction
                        and key not in self.frozen_parameters
                    ):
                        param.requires_grad_()
                    elif (
                        parameter in ("interaction", "all")
                        and interaction
                        and key not in self.frozen_parameters
                    ):
                        param.requires_grad_()
        # pylint: disable-next=consider-using-in
        if parameter != "mono" and parameter != "interaction":
            super().unfreeze(parameter)

    @override
    def update_binding_optim(
        self, binding_optim: BindingOptim
    ) -> BindingOptim:
        if len(self.symmetry) > 0:
            binding_optim.steps.append(
                Step([Call(self, "unfreeze", {"parameter": "mono"})])
            )

        # greedy search
        if self.increment_flank:
            calls: list[Call] = []
            for conv1d in self._layers:
                # pylint: disable-next=protected-access
                for bmd in set(bmd[0] for bmd in conv1d._binding_mode):
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
                    for bmd in set(bmd[0] for bmd in conv1d._binding_mode):
                        binding_optim.steps[-1].calls.append(
                            Call(
                                bmd,
                                "update_read_length",
                                {"left_shift": 1, "right_shift": 1},
                            )
                        )

        # interaction
        if self.interaction_distance > 0:
            binding_optim.steps.append(
                Step([Call(self, "unfreeze", {"parameter": "interaction"})])
            )

        binding_optim.merge_binding_optim()
        return binding_optim

    @staticmethod
    def _get_key(elements: tuple[Tensor, Tensor], index: int) -> str:
        """Key for parameter in _psam corresponding to elements of symmetry"""
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

    @override
    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        del mode
        return length - len(self.symmetry) + 1

    @override
    def in_len(self, length: T, mode: Literal["min", "max"] = "max") -> T:
        del mode
        return len(self.symmetry) + length - 1

    def get_information(self, position: int) -> float:
        with torch.inference_mode():
            psam = self.get_filter(0)
            psam = psam[: len(psam) // self.n_strands, :, position].flatten()
        probability = F.softmax(psam, dim=0)
        information = torch.log2(
            torch.tensor(probability.numel(), device=probability.device)
        )
        information += torch.sum(probability * torch.log2(probability))
        return information.item()

    def shift_heuristic(self) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        informations = [
            self.get_information(idx) for idx, _ in enumerate(self.symmetry)
        ]
        center_position = sum(
            idx * info for idx, info in enumerate(informations)
        ) / sum(informations)
        right_shift = round(center_position - ((len(self.symmetry) - 1) / 2))
        return self.update_footprint(-right_shift, right_shift)

    def update_params(
        self, di_grad: bool = False
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Update _psam according to symmetry"""
        added_params = []
        removed_keys = set(self.betas.keys())
        out_channels = self.out_channels // self.n_strands

        for dist in range(self.interaction_distance + 1):
            shape: tuple[int, ...]
            if dist == 0:
                shape = (out_channels, self.in_channels)
            else:
                shape = (out_channels, self.in_channels, self.in_channels)

            for i in range(len(self.symmetry) - dist):
                for j in range(math.prod(shape)):
                    key = self._get_key(
                        (self.symmetry[i], self.symmetry[i + dist]), j
                    )
                    if key in self.betas:
                        removed_keys.discard(key)
                    else:
                        param = torch.nn.Parameter(
                            torch.tensor(0.0, device=self.symmetry.device),
                            requires_grad=(
                                (di_grad or dist == 0)
                                and key not in self.frozen_parameters
                            ),
                        )
                        if dist == 0:
                            bound = math.sqrt(
                                1 / (self.in_channels * len(self.symmetry))
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
        """Extend or shrink the symmetry in either direction"""
        if left_shift + right_shift <= -(
            len(self.symmetry) - self.interaction_distance
        ):
            raise ValueError(
                "Cannot shrink footprint beyond the last interaction distance"
            )
        if self.max_kernel_size is not None and (
            left_shift + right_shift + len(self.symmetry)
            > self.max_kernel_size
        ):
            raise ValueError(
                f"left_shift={left_shift}, right_shift={right_shift} results "
                f"in a size of {len(self.symmetry) + left_shift + right_shift}"
                f" but max_kernel_size is {self.max_kernel_size}"
            )
        if (
            check_threshold
            and left_shift > 0
            and min(self.get_information(0), self.get_information(1))
            < self.information_threshold
        ):
            raise ValueError(
                "Cannot shift footprint left since information content is "
                f" {min(self.get_information(0), self.get_information(1))}"
                f"; {self.information_threshold} needed for footprint shift"
            )
        if (
            check_threshold
            and right_shift > 0
            and min(self.get_information(-2), self.get_information(-1))
            < self.information_threshold
        ):
            raise ValueError(
                "Cannot shift footprint right since information content is "
                f" {min(self.get_information(-2), self.get_information(-1))}"
                f"; {self.information_threshold} needed for footprint shift"
            )
        if left_shift == right_shift == 0:
            return tuple(), tuple()

        # update biases
        for conv1d in self._layers:
            # pylint: disable=protected-access
            conv1d._update_biases(
                binding_mode_left=left_shift,
                binding_mode_right=right_shift,
                window_top=-left_shift,
                window_bottom=-right_shift,
            )

        # update symmetry string
        next_val = (self.symmetry.max() + 1).unsqueeze(0)
        self.symmetry = self.symmetry[
            max(0, -left_shift) : len(self.symmetry) - max(0, -right_shift)
        ]
        for _ in range(left_shift):  # prepend
            self.symmetry = torch.cat((next_val, self.symmetry), dim=0)
            next_val += 1
        for _ in range(right_shift):  # append
            self.symmetry = torch.cat((self.symmetry, next_val), dim=0)
            next_val += 1

        # propagate updates
        for conv1d in self._layers:
            # pylint: disable=protected-access
            for bmd, layer_idx in conv1d._binding_mode:
                bmd._update_propagation(
                    layer_idx + 1,
                    left_shift=-left_shift,
                    right_shift=-right_shift,
                )
            # pylint: enable=protected-access

        # update parameters
        return self.update_params(di_grad=False)

    def _seed(
        self, seed: Sequence[Sequence[str]], seed_scale: int = 1
    ) -> None:
        """Seed mono feature vectors with given values"""

        if self.alphabet is None:
            raise ValueError("alphabet must be specified if seed is specified")
        if len(seed) != self.out_channels // self.n_strands:
            raise ValueError(
                "Need as many seeds as there are unique out_channels"
            )
        if any(len(s) != len(self.symmetry) for s in seed):
            raise ValueError("Each seed must be of length kernel_size")

        # stdev of non-seeded ~ Uniform[± 1/sqrt(in_channels * kernel_size)]
        stdev = math.sqrt(1 / (3 * self.in_channels * len(self.symmetry)))

        for out_channel, channel_seed in enumerate(seed):
            for position, seed_code in zip(self.symmetry, channel_seed):
                encoding = self.alphabet.get_encoding[seed_code]
                if position < 0:
                    encoding = tuple(
                        len(self.alphabet.alphabet) - i - 1 for i in encoding
                    )

                if len(encoding) == 0:  # don't seed
                    continue
                if len(encoding) == self.in_channels:  # zero out
                    seeded_fill = 0.0
                    unseeded_fill = 0.0
                else:
                    multiplier = math.sqrt(
                        (self.in_channels - len(encoding)) / len(encoding)
                    )
                    seeded_fill = seed_scale * stdev * multiplier
                    unseeded_fill = -seed_scale * stdev / multiplier
                    # ensure Var[seeded] = Var[non-seeded]
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

    @classmethod
    def frozen_positions(
        cls,
        positions: list[int],
        symmetry: list[int],
        in_channels: int,
        out_channels: int,
    ) -> set[str]:
        """Parameter keys to be frozen if a position on psam is frozen"""
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

    def fix_gauge(self) -> None:
        """Remove invariances between mono and interaction parameters"""
        bias_shift = 0.0
        for dist in range(1, self.interaction_distance + 1):
            for i in range(len(self.symmetry) - dist):
                feat1, feat2 = self.symmetry[i], self.symmetry[i + dist]

                di_params = [
                    self.betas[self._get_key((feat1, feat2), j)]
                    for j in range(
                        (self.out_channels // self.n_strands)
                        * self.in_channels**2
                    )
                ]
                di_param = torch.stack(cast(list[Tensor], di_params)).view(
                    (
                        self.out_channels // self.n_strands,
                        self.in_channels,
                        self.in_channels,
                    )
                )

                mono1_params = [
                    self.betas[self._get_key((feat1, feat1), j)]
                    for j in range(
                        (self.out_channels // self.n_strands)
                        * self.in_channels
                    )
                ]
                mono1_param = torch.stack(
                    cast(list[Tensor], mono1_params)
                ).view((self.out_channels // self.n_strands, self.in_channels))
                mono2_params = [
                    self.betas[self._get_key((feat2, feat2), j)]
                    for j in range(
                        (self.out_channels // self.n_strands)
                        * self.in_channels
                    )
                ]
                mono2_param = torch.stack(
                    cast(list[Tensor], mono2_params)
                ).view((self.out_channels // self.n_strands, self.in_channels))

                shift1 = di_param.mean(dim=2, keepdim=True)
                di_param -= shift1
                shift2 = di_param.mean(dim=1, keepdim=True)
                di_param -= shift2
                mono1_param += shift1.reshape(shift1.shape[0], -1)
                mono2_param += shift2.reshape(shift2.shape[0], -1)
                if self.normalize:
                    bias_shift += (
                        mono1_param.mean(-1) + mono2_param.mean(-1)
                    ).item()
                    mono1_param -= mono1_param.mean(-1)
                    mono2_param -= mono2_param.mean(-1)

                for param, val in zip(
                    di_params + mono1_params + mono2_params,
                    torch.cat(
                        [
                            i.flatten()
                            for i in (di_param, mono1_param, mono2_param)
                        ]
                    ),
                ):
                    torch.nn.init.constant_(param, val)

        self.bias.data = self.bias + bias_shift

    def get_dirichlet(self) -> Tensor:
        # combine parameters into parameter classes (all betas at a position)
        param_classes: dict[str, set[Tensor]] = {}
        for name, param in self.betas.items():
            key = name.rsplit("-", 1)[0]  # drop index from key
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

    def get_filter(self, dist: int) -> Tensor:
        """Weight matrix for a given interaction distance"""
        if not 0 <= dist <= self.interaction_distance:
            raise ValueError("dist must be in [0, self.interaction_distance]")

        params = []
        for i in range(len(self.symmetry) - dist):
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
                # autosymmetric interaction parameter
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
            # first dimension is Afor Bfor Brev Arev
            if dist == 0:
                matrix_reverse = matrix.flip(dims=[0, 1, 2])
            else:
                matrix_reverse = matrix.flip(dims=[0, 1, 2, 3]).transpose(1, 2)
            matrix = torch.cat((matrix, matrix_reverse))
        return matrix

    def get_bias(self) -> Tensor:
        if self.score_reverse:
            return torch.cat((self.bias, self.bias.flip(0)))
        return self.bias


class NonSpecific(LayerSpec):
    """Non-specific 'PSAM'"""

    def __init__(self, alphabet: Alphabet, name: str = "") -> None:
        self.alphabet = alphabet
        super().__init__(
            out_channels=1, in_channels=len(alphabet.alphabet), name=name
        )

    @override
    def out_len(
        self, length: T, mode: Literal["min", "max", "shape"] = "shape"
    ) -> T:
        del mode
        return length * 0 + 1

    @overload
    def in_len(self, length: T, mode: Literal["min"]) -> T:
        ...

    @overload
    def in_len(self, length: T, mode: Literal["max"]) -> None:
        ...

    @override
    def in_len(
        self, length: T, mode: Literal["min", "max"] = "max"
    ) -> T | None:
        if mode == "min":
            return length * 0 + 1
        return None
