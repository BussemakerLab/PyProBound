# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, missing-module-docstring, protected-access
# mypy: disable-error-code="no-untyped-call"
import unittest
from typing import cast

import torch
from torch import Tensor
from typing_extensions import override

import pyprobound

from . import make_count_table
from .test_layers import BaseTestCases


def initialize_conv1d(
    layer: pyprobound.layers.Conv1d | pyprobound.layers.Conv0d,
) -> None:
    if layer.train_posbias:
        torch.nn.init.normal_(layer.log_posbias)

    if isinstance(layer, pyprobound.layers.Conv1d):
        for param in layer.layer_spec.betas.values():
            torch.nn.init.uniform_(param, -0.5, 0.5)


class TestConv1d_4_dense(BaseTestCases.BaseTestLayer):
    layer: pyprobound.layers.Conv1d

    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                pairwise_distance=0,
                information_threshold=0.0,
            ),
            self.count_table,
            one_hot=False,
            normalize=False,
        )
        initialize_conv1d(self.layer)

    def test_embedding_size(self) -> None:
        self.layer.one_hot = False
        seq = self.count_table.seqs[0]
        dense_embedding_size = seq.numel() * seq.element_size()
        self.assertEqual(
            self.layer.max_embedding_size(),
            dense_embedding_size,
            "incorrect dense embedding size",
        )

        self.layer.one_hot = True
        alphabet = self.count_table.alphabet
        if self.layer.layer_spec.pairwise_distance > 0:
            embedded_seq = alphabet.pairwise_embedding(seq)
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
        self.layer.log_posbias.requires_grad_(False)
        self.layer.log_posbias.zero_()

        super().test_out_len_finite()

    @override
    def test_in_len(self) -> None:
        del self.count_table[100:]
        self.layer.log_posbias.requires_grad_(False)
        self.layer.log_posbias.zero_()

        super().test_in_len()

    @override
    def test_update_limit(self) -> None:
        if self.layer.bias_mode == "reverse":
            self.skipTest("Cannot test limit with reverse bias mode")
        super().test_update_limit()

    def check_nans(self, tensor: Tensor) -> None:
        self.assertFalse(
            torch.any(torch.isnan(tensor)), "output contains NaNs"
        )

    def test_dense_v_onehot(self) -> None:
        with torch.enable_grad():
            for p in self.layer.parameters():
                p.requires_grad_()
            self.layer.one_hot = False
            dense_out = self.layer(self.count_table.seqs)
            dense_out.logsumexp(tuple(range(dense_out.ndim))).backward()
            dense_grad = torch.cat(
                [
                    cast(Tensor, p.grad).flatten()
                    for p in self.layer.parameters()
                ]
            )

            self.layer.zero_grad()
            self.layer.one_hot = True
            onehot_out = self.layer(self.count_table.seqs)
            onehot_out.logsumexp(tuple(range(onehot_out.ndim))).backward()
            onehot_grad = torch.cat(
                [
                    cast(Tensor, p.grad).flatten()
                    for p in self.layer.parameters()
                ]
            )

            self.check_nans(dense_out)
            self.check_nans(onehot_out)
            self.check_nans(dense_grad)
            self.check_nans(onehot_grad)

            self.assertTrue(
                torch.allclose(dense_out, onehot_out, atol=1e-6),
                "dense and one_hot outputs do not match",
            )
            self.assertTrue(
                torch.allclose(
                    dense_grad, onehot_grad, atol=1e-6, equal_nan=True
                ),
                "dense and one_hot grads do not match",
            )

    def test_forward_v_reverse(self) -> None:
        if self.layer.bias_mode == "same" or self.layer.bias_bin != 1:
            self.layer.layer_spec._score_reverse = True
            self.layer.log_posbias.requires_grad_(False)
            self.layer.log_posbias.zero_()
        else:
            self.layer.layer_spec._score_reverse = True
            self.layer._bias_mode = "reverse"
            self.layer.log_posbias = torch.nn.Parameter(
                self.layer.log_posbias[
                    :, : self.layer.layer_spec.out_channels // 2
                ]
            )

        rev_seqs = (
            len(self.count_table.alphabet.alphabet) - 1 - self.count_table.seqs
        )
        rev_seqs[rev_seqs < 0] = self.count_table.seqs[rev_seqs < 0]
        rev_seqs = rev_seqs.flip(1)

        forward_out = self.layer(self.count_table.seqs)
        reverse_out = self.layer(rev_seqs).flip(-1, -2)

        self.check_nans(forward_out)
        self.check_nans(reverse_out)
        self.assertTrue(
            torch.allclose(forward_out, reverse_out, atol=1e-6),
            "forward and reverse outputs do not match",
        )

    def test_conv_v_unfold(self) -> None:
        with torch.enable_grad():
            for p in self.layer.parameters():
                p.requires_grad_()
            self.layer.one_hot = True
            self.layer.unfold = False
            conv1d_out = self.layer(self.count_table.seqs)
            conv1d_out.logsumexp(tuple(range(conv1d_out.ndim))).backward()
            conv1d_grad = torch.cat(
                [
                    cast(Tensor, p.grad).flatten()
                    for p in self.layer.parameters()
                ]
            )

            self.layer.zero_grad()
            self.layer.unfold = True
            unfold_out = self.layer(self.count_table.seqs)
            unfold_out.logsumexp(tuple(range(unfold_out.ndim))).backward()
            unfold_grad = torch.cat(
                [
                    cast(Tensor, p.grad).flatten()
                    for p in self.layer.parameters()
                ]
            )

            self.check_nans(conv1d_out)
            self.check_nans(unfold_out)
            self.check_nans(conv1d_grad)
            self.check_nans(unfold_grad)

            self.assertTrue(
                torch.allclose(conv1d_out, unfold_out, atol=1e-6),
                "conv1d and unfold outputs do not match",
            )
            self.assertTrue(
                torch.allclose(
                    conv1d_grad, unfold_grad, atol=1e-6, equal_nan=True
                ),
                "conv1d and unfold grads do not match",
            )

    def test_shift_footprint(self) -> None:
        del self.count_table[100:]
        shift = self.layer.bias_bin
        score_shift = shift + self.layer.layer_spec.dilation - 1
        self.layer.log_posbias[..., :score_shift] = 0
        self.layer.log_posbias[..., -score_shift:] = 0

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
                curr_score[:, 0], prev_score[:, 0, score_shift:], atol=1e-6
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
                        .roll(i[0].isneginf().sum().item())[:-score_shift]
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
        if self.layer.bias_mode == "reverse":
            self.skipTest("Cannot increase flank with reverse bias mode")
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

        prev_score = self.layer(self.count_table.seqs)
        self.layer.layer_spec.fix_gauge()
        curr_score = self.layer(self.count_table.seqs)
        self.assertTrue(
            torch.allclose(prev_score, curr_score, atol=1e-6),
            "incorrect output after fixing gauge",
        )
        for dist in range(self.layer.layer_spec.pairwise_distance + 1):
            for channel in self.layer.layer_spec.get_filter(dist):
                for pos in channel.unbind(-1):
                    mean_0 = pos.mean(dim=0)
                    self.assertTrue(
                        torch.allclose(
                            mean_0, torch.zeros_like(mean_0), atol=1e-6
                        ),
                        f"gauge fixing did not zero out dist={dist} dim=0",
                    )
                    if pos.ndim > 1:
                        mean_1 = pos.mean(dim=1)
                        self.assertTrue(
                            torch.allclose(
                                mean_1, torch.zeros_like(mean_1), atol=1e-6
                            ),
                            f"gauge fixing did not zero out dist={dist} dim=1",
                        )


class TestConv1d_4_onehot(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                pairwise_distance=0,
                information_threshold=0.0,
            ),
            self.count_table,
            one_hot=True,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4dilate2_onehot(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                pairwise_distance=0,
                information_threshold=0.0,
                dilation=2,
            ),
            self.count_table,
            one_hot=True,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4dilate3(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                pairwise_distance=0,
                information_threshold=0.0,
                dilation=3,
            ),
            self.count_table,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4di1_dense(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                pairwise_distance=1,
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
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                pairwise_distance=1,
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
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                pairwise_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            unfold=True,
            train_posbias=True,
            bias_bin=1,
            one_hot=False,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4dilate2_bin1_dense(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                dilation=2,
                information_threshold=0.0,
            ),
            self.count_table,
            train_posbias=True,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4di1_bin1nolength(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                pairwise_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            unfold=True,
            train_posbias=True,
            bias_bin=1,
            length_specific_bias=False,
            one_hot=False,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4di1_bin1same(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                pairwise_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            unfold=True,
            train_posbias=True,
            bias_bin=1,
            bias_mode="same",
            one_hot=False,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4di1_bin1reverse(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                pairwise_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            unfold=True,
            train_posbias=True,
            bias_bin=1,
            bias_mode="reverse",
            one_hot=False,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4di1_bin2_onehot(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=4,
                pairwise_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            unfold=True,
            train_posbias=True,
            bias_bin=2,
            one_hot=True,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4di1_out4(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                out_channels=4,
                kernel_size=4,
                pairwise_distance=1,
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
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                out_channels=4,
                kernel_size=4,
                pairwise_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            unfold=True,
            train_posbias=True,
            bias_bin=1,
            one_hot=True,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4di1_bin2_out4(TestConv1d_4_dense):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                out_channels=4,
                kernel_size=4,
                pairwise_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            unfold=True,
            train_posbias=True,
            bias_bin=2,
            one_hot=True,
            normalize=False,
        )
        initialize_conv1d(self.layer)


class TestConv1d_4_sharedPSAM(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.psam = pyprobound.layers.PSAM(
            alphabet=self.count_table.alphabet,
            kernel_size=4,
            information_threshold=0.0,
            normalize=False,
        )
        self.layers = [
            pyprobound.layers.Conv1d.from_psam(
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
    layer: pyprobound.layers.Conv0d

    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(n_seqs=65537)
        self.layer = pyprobound.layers.Conv0d.from_nonspecific(
            pyprobound.layers.NonSpecific(alphabet=self.count_table.alphabet),
            self.count_table,
            train_posbias=True,
        )
        initialize_conv1d(self.layer)

    @override
    def test_out_len_finite(self) -> None:
        self.layer.log_posbias.requires_grad_(False)
        self.layer.log_posbias.zero_()
        super().test_out_len_finite()

    @override
    def test_in_len(self) -> None:
        self.layer.log_posbias.requires_grad_(False)
        self.layer.log_posbias.zero_()
        super().test_in_len()


if __name__ == "__main__":
    unittest.main()
