# pylint: disable=invalid-name, missing-class-docstring, missing-module-docstring, protected-access
import unittest

import torch
from torch import Tensor
from typing_extensions import override

import probound.conv1d
import probound.psam

from . import make_count_table
from .test_layers import BaseTestCases


def initialize_conv1d(
    layer: probound.conv1d.Conv1d | probound.conv1d.Conv0d,
) -> None:
    if layer.train_omega:
        torch.nn.init.normal_(layer.omega)

    if isinstance(layer, probound.conv1d.Conv1d):
        for param in layer.layer_spec.betas.values():
            torch.nn.init.uniform_(param, -0.5, 0.5)

        if layer.train_theta:
            torch.nn.init.normal_(layer.theta)


class TestConv1d_4_dense(BaseTestCases.BaseTestLayer):
    layer: probound.conv1d.Conv1d

    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                interaction_distance=0,
                information_threshold=0.0,
            ),
            self.count_table,
            one_hot=False,
            normalize=False,
        )
        initialize_conv1d(self.layer)

    def test_embedding_size(self) -> None:
        self.layer._one_hot = torch.tensor(False)
        seq = self.count_table.seqs[0]
        dense_embedding_size = seq.numel() * seq.element_size()
        self.assertEqual(
            self.layer.max_embedding_size(),
            dense_embedding_size,
            "incorrect dense embedding size",
        )

        self.layer._one_hot = torch.tensor(True)
        alphabet = self.count_table.alphabet
        if self.layer.layer_spec.interaction_distance > 0:
            embedded_seq = alphabet.interaction_embedding(seq)
        else:
            embedded_seq = alphabet.embedding(seq)
        onehot_embedding_size = (
            embedded_seq.numel() * embedded_seq.element_size()
        )
        self.assertEqual(
            self.layer.max_embedding_size(),
            onehot_embedding_size,
            "incorrect onehot embedding size",
        )

    @override
    def test_out_len_finite(self) -> None:
        del self.count_table[100:]
        self.layer.omega.requires_grad_(False)
        self.layer.omega.zero_()
        self.layer.theta.requires_grad_(False)
        self.layer.theta.zero_()

        super().test_out_len_finite()

    @override
    def test_in_len(self) -> None:
        del self.count_table[100:]
        self.layer.omega.requires_grad_(False)
        self.layer.omega.zero_()
        self.layer.theta.requires_grad_(False)
        self.layer.theta.zero_()

        super().test_in_len()

    def check_nans(self, tensor: Tensor) -> None:
        self.assertFalse(
            torch.any(torch.isnan(tensor)), "output contains NaNs"
        )

    def test_dense_v_onehot(self) -> None:
        self.layer._one_hot = torch.tensor(False)
        dense_out = self.layer(self.count_table.seqs)
        self.layer._one_hot = torch.tensor(True)
        onehot_out = self.layer(self.count_table.seqs)
        self.check_nans(dense_out)
        self.check_nans(onehot_out)
        self.assertTrue(
            torch.allclose(dense_out, onehot_out, atol=1e-6),
            "dense and one_hot outputs do not match",
        )

    def test_forward_v_reverse(self) -> None:
        self.layer.layer_spec._score_reverse = torch.tensor(True)
        self.layer.omega.requires_grad_(False)
        self.layer.omega.zero_()
        self.layer.theta.requires_grad_(False)
        self.layer.theta.zero_()

        rev_seqs = torch.stack(
            [
                torch.roll(
                    torch.flip(
                        len(self.count_table.alphabet.alphabet) - 1 - i,
                        dims=(0,),
                    ),
                    shifts=-(i == self.count_table.alphabet.neginf_pad)
                    .sum()
                    .item(),
                )
                for i in self.count_table.seqs
            ]
        )
        rev_seqs[rev_seqs < 0] = self.count_table.alphabet.neginf_pad

        forward_out = self.layer(self.count_table.seqs)
        reverse_out = self.layer(rev_seqs)
        flipped_reverse_out = torch.stack(
            [
                torch.roll(i.flip(0, 1), -i[0].isneginf().sum().item())
                for i in reverse_out
            ]
        )

        self.check_nans(forward_out)
        self.check_nans(flipped_reverse_out)
        self.assertTrue(
            torch.allclose(forward_out, flipped_reverse_out, atol=1e-6),
            "forward and reverse outputs do not match",
        )

    def test_conv_v_unfold(self) -> None:
        self.layer._one_hot = torch.tensor(True)
        torch.nn.init.zeros_(self.layer.theta)

        # test conv
        self.layer.theta.requires_grad_(False)
        conv1d_out = self.layer(self.count_table.seqs)
        self.layer.theta.requires_grad_(True)
        unfold_out = self.layer(self.count_table.seqs)
        self.check_nans(conv1d_out)
        self.check_nans(unfold_out)
        self.assertTrue(
            torch.allclose(conv1d_out, unfold_out, atol=1e-6),
            "unfold output does not match conv1d output",
        )

    def test_shift_footprint(self) -> None:
        del self.count_table[100:]
        shift = self.layer.bias_bin
        self.layer.theta[..., :shift, :] = 0
        self.layer.theta[..., -shift:, :] = 0
        self.layer.omega[..., :shift] = 0
        self.layer.omega[..., -shift:] = 0

        # increment symmetry left
        prev_score = self.layer(self.count_table.seqs)
        added_params = self.layer.layer_spec.update_footprint(
            left_shift=shift
        )[0]
        for param in added_params:
            param.detach().zero_()
        curr_score = self.layer(self.count_table.seqs)
        self.check_nans(curr_score)
        self.check_nans(prev_score)
        self.layer.check_length_consistency()
        self.assertTrue(
            torch.allclose(
                curr_score[:, 0], prev_score[:, 0, shift:], atol=1e-6
            ),
            "incorrect output after increasing symmetry left",
        )

        # decrement symmetry left
        self.layer.layer_spec.update_footprint(left_shift=-shift)
        curr_score = self.layer(self.count_table.seqs)
        self.check_nans(curr_score)
        self.layer.check_length_consistency()
        self.assertTrue(
            torch.allclose(curr_score, prev_score, atol=1e-6),
            "incorrect output after decrementing symmetry left",
        )

        # increment symmetry right
        prev_score = self.layer(self.count_table.seqs)
        added_params = self.layer.layer_spec.update_footprint(
            right_shift=shift
        )[0]
        for param in added_params:
            param.detach().zero_()
        curr_score = self.layer(self.count_table.seqs)
        self.check_nans(curr_score)
        self.check_nans(prev_score)
        self.layer.check_length_consistency()
        self.assertTrue(
            torch.allclose(
                curr_score[:, 0],
                torch.stack(
                    [
                        i[0]
                        .roll(i[0].isneginf().sum().item())[:-shift]
                        .roll(-i[0].isneginf().sum().item())
                        for i in prev_score
                    ]
                ),
                atol=1e-6,
            ),
            "incorrect output after increasing symmetry right",
        )

        # decrement symmetry right
        self.layer.layer_spec.update_footprint(right_shift=-shift)
        curr_score = self.layer(self.count_table.seqs)
        self.check_nans(curr_score)
        self.layer.check_length_consistency()
        self.assertTrue(
            torch.allclose(curr_score, prev_score, atol=1e-6),
            "incorrect output after decrementing symmetry right",
        )

    def test_increase_flank(self) -> None:
        del self.count_table[100:]
        shift = self.layer.bias_bin
        prev_score = self.layer(self.count_table.seqs)
        self.check_nans(prev_score)

        # increment left flank
        self.count_table.set_flank_length(left=shift)
        self.layer.update_input_length(left_shift=shift)
        self.layer.check_length_consistency()
        curr_score = self.layer(self.count_table.seqs)
        self.check_nans(curr_score)
        self.assertTrue(
            torch.allclose(prev_score, curr_score[:, :, shift:], atol=1e-6),
            "incorrect output after increasing left flank",
        )

        # decrement left flank
        self.count_table.set_flank_length(0, 0)
        self.layer.update_input_length(left_shift=-shift)
        self.layer.check_length_consistency()
        curr_score = self.layer(self.count_table.seqs)
        self.check_nans(curr_score)
        self.assertTrue(
            torch.allclose(prev_score, curr_score, atol=1e-6),
            "incorrect output after resetting left flank",
        )

        # increment right flank
        self.count_table.set_flank_length(right=shift)
        self.layer.update_input_length(right_shift=shift)
        self.layer.check_length_consistency()
        curr_score = self.layer(self.count_table.seqs)
        self.check_nans(curr_score)
        update_window = self.layer.out_len(self.count_table.variable_lengths)
        self.assertTrue(
            torch.allclose(
                prev_score,
                torch.stack(
                    [
                        torch.cat(
                            [
                                curr_score[i, :, : update_window[i]],
                                curr_score[i, :, update_window[i] + shift :],
                            ],
                            dim=1,
                        )
                        for i in range(len(self.count_table))
                    ]
                ),
                atol=1e-6,
            ),
            "incorrect output after increasing right flank",
        )

        # decrement right flank
        self.count_table.set_flank_length(0, 0)
        self.layer.update_input_length(right_shift=-shift)
        self.layer.check_length_consistency()
        curr_score = self.layer(self.count_table.seqs)
        self.check_nans(curr_score)
        self.assertTrue(
            torch.allclose(prev_score, curr_score, atol=1e-6),
            "incorrect output after resetting right flank",
        )

    def test_fix_gauge(self) -> None:
        del self.count_table[100:]
        self.layer.theta.requires_grad_(False)
        self.layer.theta.zero_()

        prev_score = self.layer(self.count_table.seqs)
        self.layer.layer_spec.fix_gauge()
        curr_score = self.layer(self.count_table.seqs)
        self.assertTrue(
            torch.allclose(prev_score, curr_score, atol=1e-6),
            "incorrect output after fixing gauge",
        )
        for dist in range(1, self.layer.layer_spec.interaction_distance + 1):
            for channel in self.layer.layer_spec.get_filter(dist):
                for pos in channel.transpose(0, -1):
                    mean_0 = pos.mean(dim=0)
                    mean_1 = pos.mean(dim=1)
                    self.assertTrue(
                        torch.allclose(
                            mean_0, torch.zeros(len(mean_0)), atol=1e-6
                        ),
                        "gauge fixing did not zero out average of di dim=0",
                    )
                    self.assertTrue(
                        torch.allclose(
                            mean_1, torch.zeros(len(mean_1)), atol=1e-6
                        ),
                        "gauge fixing did not zero out average of di dim=1",
                    )


class TestConv1d_4_onehot(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                interaction_distance=0,
                information_threshold=0.0,
            ),
            self.count_table,
            one_hot=True,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4di1_dense(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            one_hot=False,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4di1_onehot(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            one_hot=True,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4di1_bin1_dense(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            train_theta=True,
            train_omega=True,
            bias_bin=1,
            one_hot=False,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4di1_bin2_onehot(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            train_theta=True,
            train_omega=True,
            bias_bin=2,
            one_hot=True,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4di1_out4(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                out_channels=4,
                kernel_size=4,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            one_hot=True,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4di1_bin1_out4(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                out_channels=4,
                kernel_size=4,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            train_theta=True,
            train_omega=True,
            bias_bin=1,
            one_hot=True,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4di1_bin2_out4(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                out_channels=4,
                kernel_size=4,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            train_theta=True,
            train_omega=True,
            bias_bin=2,
            one_hot=True,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4_sharedPSAM(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.psam = probound.psam.PSAM(
            alphabet=self.count_table.alphabet,
            kernel_size=4,
            information_threshold=0.0,
            normalize=False,
        )
        self.layers = [
            probound.conv1d.Conv1d.from_psam(
                self.psam, self.count_table, normalize=False
            )
            for _ in range(2)
        ]
        for layer in self.layers:
            initialize_conv1d(layer)

    def test_shift_shared_footprint(self) -> None:
        for layer in self.layers:
            layer.check_length_consistency()
        self.psam.update_footprint(left_shift=1)
        for layer in self.layers:
            layer.check_length_consistency()
        self.psam.update_footprint(right_shift=1)
        for layer in self.layers:
            layer.check_length_consistency()
        self.psam.update_footprint(left_shift=-1, right_shift=-1)
        for layer in self.layers:
            layer.check_length_consistency()


class TestConv0d(BaseTestCases.BaseTestLayer):
    layer: probound.conv1d.Conv0d

    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = probound.conv1d.Conv0d.from_nonspecific(
            probound.psam.NonSpecific(alphabet=self.count_table.alphabet),
            self.count_table,
            train_omega=True,
        )
        initialize_conv1d(self.layer)

    @override
    def test_out_len_finite(self) -> None:
        self.layer.omega.requires_grad_(False)
        self.layer.omega.zero_()
        super().test_out_len_finite()

    @override
    def test_in_len(self) -> None:
        self.layer.omega.requires_grad_(False)
        self.layer.omega.zero_()
        super().test_in_len()


if __name__ == "__main__":
    unittest.main()
