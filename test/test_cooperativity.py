# pylint: disable=invalid-name, missing-class-docstring, missing-module-docstring
import unittest

import torch
from torch import Tensor
from typing_extensions import override

import probound.binding
import probound.conv1d
import probound.cooperativity
import probound.layers
import probound.psam

from . import make_count_table
from .test_binding import initialize_binding


def initialize_cooperativity(
    binding_cooperativity: probound.cooperativity.BindingCooperativity,
) -> None:
    torch.nn.init.normal_(binding_cooperativity.spacing.flattened_spacing)
    if binding_cooperativity.train_omega:
        torch.nn.init.normal_(binding_cooperativity.omega)
    else:
        binding_cooperativity.omega.requires_grad_(False)
    for bmd in (
        binding_cooperativity.binding_mode_0,
        binding_cooperativity.binding_mode_1,
    ):
        initialize_binding(bmd)


class TestCooperativity_5__3(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_0 = probound.binding.BindingMode(
            [
                probound.conv1d.Conv1d.from_psam(
                    probound.psam.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=5,
                        information_threshold=0.0,
                        normalize=False,
                    ),
                    self.count_table,
                    normalize=False,
                )
            ]
        )
        binding_mode_1 = probound.binding.BindingMode(
            [
                probound.conv1d.Conv1d.from_psam(
                    probound.psam.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=3,
                        information_threshold=0.0,
                        normalize=False,
                    ),
                    self.count_table,
                    normalize=False,
                )
            ]
        )
        self.binding_cooperativity = (
            probound.cooperativity.BindingCooperativity(
                probound.cooperativity.Spacing(
                    binding_mode_0.key(), binding_mode_1.key(), normalize=False
                ),
                binding_mode_0,
                binding_mode_1,
                normalize=False,
            )
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()

    def test_embedding_size(self) -> None:
        windows = self.binding_cooperativity.score_windows(
            self.count_table.seqs[:1]
        )
        expected_embedding_size = windows.numel() * windows.element_size()
        expected_embedding_size = max(
            self.binding_cooperativity.binding_mode_0.max_embedding_size(),
            self.binding_cooperativity.binding_mode_1.max_embedding_size(),
            expected_embedding_size,
        )
        self.assertEqual(
            self.binding_cooperativity.max_embedding_size(),
            expected_embedding_size,
            "incorrect embedding size",
        )

    def test_spacing_gauge(self) -> None:
        spacing = self.binding_cooperativity.spacing
        zero_gauge: Tensor | None = None
        for i in range(1, spacing.max_num_windows + 1):
            for j in range(1, spacing.max_num_windows + 1):
                if zero_gauge is None:
                    zero_gauge = spacing.get_spacing(i, j)[0, 0]
                self.assertEqual(
                    zero_gauge,
                    spacing.get_spacing(i, j)[0, 0],
                    "Spacing main diagonal inconsistent with changing windows",
                )

    def test_spacing(self) -> None:
        prev_omega = self.binding_cooperativity.omega
        self.binding_cooperativity.omega.requires_grad_(False)
        torch.nn.init.zeros_(self.binding_cooperativity.omega)

        windows_0 = self.binding_cooperativity.binding_mode_0.score_windows(
            self.count_table.seqs
        )
        windows_1 = self.binding_cooperativity.binding_mode_1.score_windows(
            self.count_table.seqs
        )
        with (
            unittest.mock.patch.object(
                self.binding_cooperativity.binding_mode_0, "score_windows"
            ) as mock_windows_0,
            unittest.mock.patch.object(
                self.binding_cooperativity.binding_mode_1, "score_windows"
            ) as mock_windows_1,
        ):
            for offset in range(-windows_0.shape[-1] + 1, windows_1.shape[-1]):
                expected_output = None
                for i in range(windows_0.shape[-1]):
                    j = i + offset
                    if j < 0 or j >= windows_1.shape[-1]:
                        continue
                    mock_windows_0_return = torch.full_like(
                        windows_0[:1], float("-inf")
                    )
                    mock_windows_1_return = torch.full_like(
                        windows_1[:1], float("-inf")
                    )
                    mock_windows_0_return[..., i] = 0
                    mock_windows_1_return[..., j] = 0
                    mock_windows_0.return_value = mock_windows_0_return
                    mock_windows_1.return_value = mock_windows_1_return
                    output = self.binding_cooperativity(self.count_table.seqs)
                    if expected_output is None:
                        expected_output = output
                    self.assertTrue(
                        torch.allclose(output, expected_output),
                        "BindingCooperativity not scoring spacings equally",
                    )

        self.binding_cooperativity.omega = prev_omega

    def check_nans(self, tensor: Tensor) -> None:
        self.assertFalse(
            torch.any(torch.isnan(tensor)), "output contains NaNs"
        )

    def test_shift_footprint(self) -> None:
        n_iter = 3
        for bmd in self.binding_cooperativity.components():
            for layer in bmd.layers:
                if not isinstance(layer, probound.conv1d.Conv1d):
                    continue
                self.binding_cooperativity.check_length_consistency()
                self.check_nans(
                    self.binding_cooperativity(self.count_table.seqs)
                )

                for _ in range(n_iter):
                    layer.layer_spec.update_footprint(left_shift=1)
                    self.binding_cooperativity.check_length_consistency()
                    self.check_nans(
                        self.binding_cooperativity(self.count_table.seqs)
                    )

                layer.layer_spec.update_footprint(left_shift=-n_iter)
                self.binding_cooperativity.check_length_consistency()
                self.check_nans(
                    self.binding_cooperativity(self.count_table.seqs)
                )

                for _ in range(n_iter):
                    layer.layer_spec.update_footprint(right_shift=1)
                    self.binding_cooperativity.check_length_consistency()
                    self.check_nans(
                        self.binding_cooperativity(self.count_table.seqs)
                    )

                layer.layer_spec.update_footprint(right_shift=-n_iter)
                self.binding_cooperativity.check_length_consistency()
                self.check_nans(
                    self.binding_cooperativity(self.count_table.seqs)
                )

    def test_increase_flank(self) -> None:
        n_iter = 5
        for i in range(1, n_iter + 1):
            self.count_table.set_flank_length(left=i, right=i - 1)
            for bmd in self.binding_cooperativity.components():
                bmd.update_read_length(left_shift=1)
            self.binding_cooperativity.check_length_consistency()
            self.check_nans(self.binding_cooperativity(self.count_table.seqs))

            self.count_table.set_flank_length(left=i, right=i)
            for bmd in self.binding_cooperativity.components():
                bmd.update_read_length(right_shift=1)
            self.binding_cooperativity.check_length_consistency()
            self.check_nans(self.binding_cooperativity(self.count_table.seqs))

        self.count_table.set_flank_length(left=0, right=0)
        for bmd in self.binding_cooperativity.components():
            bmd.update_read_length(left_shift=-n_iter, right_shift=-n_iter)
        self.binding_cooperativity.check_length_consistency()
        self.check_nans(self.binding_cooperativity(self.count_table.seqs))

    def test_shift_spacing(self) -> None:
        if not all(
            all(isinstance(i, probound.conv1d.Conv1d) for i in bmd.layers)
            for bmd in self.binding_cooperativity.components()
        ):
            self.skipTest("Can't test spacing if any non-Conv1d layers")

        # store and strip out length information
        info: list[dict[str, torch.nn.Parameter]] = []
        for bmd in self.binding_cooperativity.components():
            for layer in bmd.layers:
                if not isinstance(layer, probound.conv1d.Conv1d):
                    info.append({})
                    continue

                info.append(
                    {"prev_theta": layer.theta, "prev_omega": layer.omega}
                )

                layer.theta.requires_grad_(False)
                layer.theta.zero_()
                layer.omega.requires_grad_(False)
                layer.omega.zero_()

        # establish baseline
        prev_score = self.binding_cooperativity(self.count_table.seqs)
        self.check_nans(prev_score)
        shift = self.binding_cooperativity.bias_bin

        # shift left
        self.count_table.set_flank_length(left=shift, right=0)
        for bmd in self.binding_cooperativity.components():
            bmd.update_read_length(left_shift=shift)
        self.binding_cooperativity.omega[..., 0] = float("-inf")
        self.binding_cooperativity.omega[..., 0, :] = float("-inf")
        curr_score = self.binding_cooperativity(self.count_table.seqs)
        self.assertTrue(
            torch.allclose(curr_score, prev_score, atol=1e-6),
            "incorrect output after increasing left flank",
        )

        # reset
        self.count_table.set_flank_length(left=0, right=0)
        for bmd in self.binding_cooperativity.components():
            bmd.update_read_length(left_shift=-shift)
        curr_score = self.binding_cooperativity(self.count_table.seqs)
        self.assertTrue(
            torch.allclose(curr_score, prev_score),
            "incorrect output after increasing left flank",
        )

        # restore conv1d attributes
        for bmd in self.binding_cooperativity.components():
            for layer, layer_info in zip(bmd.layers, info):
                if not isinstance(layer, probound.conv1d.Conv1d):
                    continue
                layer.theta = layer_info["prev_theta"]
                layer.omega = layer_info["prev_omega"]


class TestCooperativity_5__3__norev(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_0 = probound.binding.BindingMode(
            [
                probound.conv1d.Conv1d.from_psam(
                    probound.psam.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=5,
                        score_reverse=False,
                        information_threshold=0.0,
                        normalize=False,
                    ),
                    self.count_table,
                    normalize=False,
                )
            ]
        )
        binding_mode_1 = probound.binding.BindingMode(
            [
                probound.conv1d.Conv1d.from_psam(
                    probound.psam.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=3,
                        score_reverse=False,
                        information_threshold=0.0,
                        normalize=False,
                    ),
                    self.count_table,
                    normalize=False,
                )
            ]
        )
        self.binding_cooperativity = (
            probound.cooperativity.BindingCooperativity(
                probound.cooperativity.Spacing(
                    binding_mode_0.key(), binding_mode_1.key(), normalize=False
                ),
                binding_mode_0,
                binding_mode_1,
                normalize=False,
            )
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__omega(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_0 = probound.binding.BindingMode(
            [
                probound.conv1d.Conv1d.from_psam(
                    probound.psam.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=5,
                        information_threshold=0.0,
                        normalize=False,
                    ),
                    self.count_table,
                    normalize=False,
                )
            ]
        )
        binding_mode_1 = probound.binding.BindingMode(
            [
                probound.conv1d.Conv1d.from_psam(
                    probound.psam.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=3,
                        information_threshold=0.0,
                        normalize=False,
                    ),
                    self.count_table,
                    normalize=False,
                )
            ]
        )
        self.binding_cooperativity = (
            probound.cooperativity.BindingCooperativity(
                probound.cooperativity.Spacing(
                    binding_mode_0.key(), binding_mode_1.key(), normalize=False
                ),
                binding_mode_0,
                binding_mode_1,
                train_omega=True,
                normalize=False,
            )
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__omega_norev(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_0 = probound.binding.BindingMode(
            [
                probound.conv1d.Conv1d.from_psam(
                    probound.psam.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=5,
                        score_reverse=False,
                        information_threshold=0.0,
                        normalize=False,
                    ),
                    self.count_table,
                    normalize=False,
                )
            ]
        )
        binding_mode_1 = probound.binding.BindingMode(
            [
                probound.conv1d.Conv1d.from_psam(
                    probound.psam.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=3,
                        score_reverse=False,
                        information_threshold=0.0,
                        normalize=False,
                    ),
                    self.count_table,
                    normalize=False,
                )
            ]
        )
        self.binding_cooperativity = (
            probound.cooperativity.BindingCooperativity(
                probound.cooperativity.Spacing(
                    binding_mode_0.key(), binding_mode_1.key(), normalize=False
                ),
                binding_mode_0,
                binding_mode_1,
                train_omega=True,
                normalize=False,
            )
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__omega_over2_spacing5(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_0 = probound.binding.BindingMode(
            [
                probound.conv1d.Conv1d.from_psam(
                    probound.psam.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=5,
                        information_threshold=0.0,
                        normalize=False,
                    ),
                    self.count_table,
                    normalize=False,
                )
            ]
        )
        binding_mode_1 = probound.binding.BindingMode(
            [
                probound.conv1d.Conv1d.from_psam(
                    probound.psam.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=3,
                        information_threshold=0.0,
                        normalize=False,
                    ),
                    self.count_table,
                    normalize=False,
                )
            ]
        )
        self.binding_cooperativity = (
            probound.cooperativity.BindingCooperativity(
                probound.cooperativity.Spacing(
                    binding_mode_0.key(),
                    binding_mode_1.key(),
                    max_overlap=2,
                    max_spacing=5,
                    normalize=False,
                ),
                binding_mode_0,
                binding_mode_1,
                train_omega=True,
                normalize=False,
            )
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_3_3_bin1__3_3_bin2(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        bmd0_l1 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            train_omega=True,
            train_theta=True,
            bias_bin=1,
            normalize=False,
        )
        bmd0_l2 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                kernel_size=3,
                in_channels=2,
                information_threshold=0.0,
                normalize=False,
            ),
            bmd0_l1,
            train_omega=True,
            train_theta=True,
            bias_bin=1,
            one_hot=True,
            normalize=False,
        )
        bmd1_l1 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            normalize=False,
        )
        bmd1_l2 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                kernel_size=3,
                in_channels=2,
                information_threshold=0.0,
                normalize=False,
            ),
            bmd1_l1,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
            normalize=False,
        )
        binding_mode_0 = probound.binding.BindingMode([bmd0_l1, bmd0_l2])
        binding_mode_1 = probound.binding.BindingMode([bmd1_l1, bmd1_l2])
        self.binding_cooperativity = (
            probound.cooperativity.BindingCooperativity(
                probound.cooperativity.Spacing(
                    binding_mode_0.key(), binding_mode_1.key(), normalize=False
                ),
                binding_mode_0,
                binding_mode_1,
                normalize=False,
            )
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_3_mp2floor_3__3_3mp2ceil_3(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        bmd0_l1 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            train_omega=True,
            train_theta=True,
            normalize=False,
        )
        bmd0_l2 = probound.layers.MaxPool1d.from_spec(
            probound.layers.MaxPool1dSpec(
                in_channels=bmd0_l1.out_channels,
                kernel_size=2,
                ceil_mode=False,
            ),
            bmd0_l1,
        )
        bmd0_l3 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                kernel_size=3,
                in_channels=2,
                information_threshold=0.0,
                normalize=False,
            ),
            bmd0_l2,
            train_omega=True,
            train_theta=True,
            one_hot=True,
            normalize=False,
        )
        bmd1_l1 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            train_omega=True,
            train_theta=True,
            normalize=False,
        )
        bmd1_l2 = probound.layers.MaxPool1d.from_spec(
            probound.layers.MaxPool1dSpec(
                in_channels=bmd1_l1.out_channels, kernel_size=2, ceil_mode=True
            ),
            bmd1_l1,
        )
        bmd1_l3 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                kernel_size=3,
                in_channels=2,
                information_threshold=0.0,
                normalize=False,
            ),
            bmd1_l2,
            train_omega=True,
            train_theta=True,
            one_hot=True,
            normalize=False,
        )
        binding_mode_0 = probound.binding.BindingMode(
            [bmd0_l1, bmd0_l2, bmd0_l3]
        )
        binding_mode_1 = probound.binding.BindingMode(
            [bmd1_l1, bmd1_l2, bmd1_l3]
        )
        self.binding_cooperativity = (
            probound.cooperativity.BindingCooperativity(
                probound.cooperativity.Spacing(
                    binding_mode_0.key(), binding_mode_1.key(), normalize=False
                ),
                binding_mode_0,
                binding_mode_1,
                normalize=False,
            )
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3_sharedSpacing(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.psam_1 = probound.psam.PSAM(
            alphabet=self.count_table.alphabet,
            kernel_size=5,
            information_threshold=0.0,
            normalize=False,
        )
        self.psam_2 = probound.psam.PSAM(
            alphabet=self.count_table.alphabet,
            kernel_size=3,
            information_threshold=0.0,
            normalize=False,
        )
        self.spacing = probound.cooperativity.Spacing.from_psams(
            self.psam_1, self.psam_2
        )
        self.coops = [
            probound.cooperativity.BindingCooperativity(
                self.spacing,
                probound.binding.BindingMode(
                    [
                        probound.conv1d.Conv1d.from_psam(
                            self.psam_1, self.count_table, normalize=False
                        )
                    ]
                ),
                probound.binding.BindingMode(
                    [
                        probound.conv1d.Conv1d.from_psam(
                            self.psam_2, self.count_table, normalize=False
                        )
                    ]
                ),
            )
            for _ in range(2)
        ]

        for coop in self.coops:
            initialize_cooperativity(coop)
            coop.check_length_consistency()

    def test_shift_shared_footprint(self) -> None:
        for coop in self.coops:
            coop.check_length_consistency()
        self.psam_1.update_footprint(left_shift=1)
        for coop in self.coops:
            coop.check_length_consistency()
        self.psam_1.update_footprint(right_shift=1)
        for coop in self.coops:
            coop.check_length_consistency()
        self.psam_1.update_footprint(left_shift=-1, right_shift=-1)
        for coop in self.coops:
            coop.check_length_consistency()
        self.psam_2.update_footprint(left_shift=1)
        for coop in self.coops:
            coop.check_length_consistency()
        self.psam_2.update_footprint(right_shift=1)
        for coop in self.coops:
            coop.check_length_consistency()
        self.psam_2.update_footprint(left_shift=-1, right_shift=-1)
        for coop in self.coops:
            coop.check_length_consistency()


if __name__ == "__main__":
    unittest.main()
