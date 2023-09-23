# pylint: disable=invalid-name, missing-class-docstring, protected-access, missing-module-docstring
import unittest

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

import probound.binding
import probound.conv1d
import probound.layers
import probound.psam

from . import make_count_table
from .test_conv1d import initialize_conv1d


def initialize_binding(binding: probound.binding.BindingMode) -> None:
    for layer in binding.layers:
        if isinstance(layer, probound.conv1d.Conv1d):
            initialize_conv1d(layer)


class TestBindingMode_0(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.binding = probound.binding.BindingMode(
            [
                probound.conv1d.Conv0d.from_nonspecific(
                    probound.psam.NonSpecific(
                        alphabet=self.count_table.alphabet
                    ),
                    self.count_table,
                    train_omega=True,
                )
            ]
        )
        initialize_binding(self.binding)

    def test_out_len_shape(self) -> None:
        seqs = self.count_table.seqs
        out = self.binding.score_windows(seqs)
        self.assertEqual(
            out.shape,
            out.shape[:-1]
            + (self.binding.out_len(seqs.shape[-1], mode="shape"),),
            "incorrect out_len in mode='shape'",
        )

    def test_out_len_finite(self) -> None:
        # store and strip out length information
        buffers: list[dict[str, Tensor]] = []
        parameters: list[dict[str, torch.nn.Parameter]] = []
        for layer in self.binding.layers:
            if not isinstance(layer, probound.conv1d.Conv1d):
                buffers.append({})
                parameters.append({})
                continue

            buffers.append(
                {
                    "prev_min_len": layer._min_input_length,
                    "prev_max_len": layer._max_input_length,
                }
            )
            parameters.append(
                {"prev_theta": layer.theta, "prev_omega": layer.omega}
            )

            layer._min_input_length = torch.tensor(float("-inf"))
            layer._max_input_length = torch.tensor(float("inf"))
            layer.theta.requires_grad_(False)
            layer.theta.zero_()
            layer.omega.requires_grad_(False)
            layer.omega.zero_()

        # get padding length
        pad_len = self.binding.in_len(1, "max")
        if pad_len is None:
            pad_len = self.binding.in_len(1, "min")

        # get out_len output
        lengths = self.binding.layers[0].lengths(self.count_table.seqs)
        out_len_min = self.binding.out_len(lengths, mode="min")
        out_len_max = self.binding.out_len(lengths, mode="max")

        # test output
        min_len = torch.tensor(float("nan"))
        max_len = torch.tensor(float("nan"))
        for i in range(pad_len + 1):
            padding_idx: float = self.count_table.alphabet.neginf_pad
            if self.count_table.seqs.ndim == 3:
                padding_idx = float("-inf")
            seqs = F.pad(
                self.count_table.seqs, (i, pad_len), value=padding_idx
            )
            output = self.binding.score_windows(seqs).isfinite().sum(-1)
            if min_len.ndim == 0:
                min_len = output
                max_len = output
            else:
                min_len = torch.minimum(min_len, output)
                max_len = torch.maximum(max_len, output)

        self.assertTrue(
            torch.equal(
                min_len,
                out_len_min.unsqueeze(-1).expand(-1, min_len.shape[-1]),
            ),
            "incorrect out_len in mode='min'",
        )
        self.assertTrue(
            torch.equal(
                max_len,
                out_len_max.unsqueeze(-1).expand(-1, max_len.shape[-1]),
            ),
            "incorrect out_len in mode='max'",
        )

        # restore conv1d attributes
        for layer, bufs, params in zip(
            self.binding.layers, buffers, parameters
        ):
            if not isinstance(layer, probound.conv1d.Conv1d):
                continue
            layer._min_input_length = bufs["prev_min_len"]
            layer._max_input_length = bufs["prev_max_len"]
            layer.theta = params["prev_theta"]
            layer.omega = params["prev_omega"]

    def test_in_len(self) -> None:
        # store and strip out length information
        buffers: list[dict[str, Tensor]] = []
        parameters: list[dict[str, torch.nn.Parameter]] = []
        for layer in self.binding.layers:
            if not isinstance(layer, probound.conv1d.Conv1d):
                buffers.append({})
                parameters.append({})
                continue

            buffers.append(
                {
                    "prev_min_len": layer._min_input_length,
                    "prev_max_len": layer._max_input_length,
                }
            )
            parameters.append(
                {"prev_theta": layer.theta, "prev_omega": layer.omega}
            )

            layer._min_input_length = torch.tensor(float("-inf"))
            layer._max_input_length = torch.tensor(float("inf"))
            layer.theta.requires_grad_(False)
            layer.theta.zero_()
            layer.omega.requires_grad_(False)
            layer.omega.zero_()

        for out_len in range(2, 5):
            # get in_len
            min_len = self.binding.in_len(out_len, "min")
            max_len = self.binding.in_len(out_len, "max")
            if max_len is None:
                self.skipTest("Can't test in_len if output is undefined")

            # test in_len predictions
            self.assertEqual(
                self.binding.score_windows(
                    self.count_table.seqs[..., : min_len - 1]
                ).shape[-1],
                out_len - 1,
                "size('min') output incorrect",
            )
            self.assertEqual(
                self.binding.score_windows(
                    self.count_table.seqs[..., :min_len]
                ).shape[-1],
                out_len,
                "size('min') output incorrect",
            )
            self.assertEqual(
                self.binding.score_windows(
                    self.count_table.seqs[..., :max_len]
                ).shape[-1],
                out_len,
                "size('max') output incorrect",
            )
            self.assertEqual(
                self.binding.score_windows(
                    self.count_table.seqs[..., : max_len + 1]
                ).shape[-1],
                out_len + 1,
                "size('max') output incorrect",
            )

        # restore conv1d attributes
        for layer, bufs, params in zip(
            self.binding.layers, buffers, parameters
        ):
            if not isinstance(layer, probound.conv1d.Conv1d):
                continue
            layer._min_input_length = bufs["prev_min_len"]
            layer._max_input_length = bufs["prev_max_len"]
            layer.theta = params["prev_theta"]
            layer.omega = params["prev_omega"]

    def check_nans(self, tensor: Tensor) -> None:
        self.assertFalse(
            torch.any(torch.isnan(tensor)), "output contains NaNs"
        )

    def test_shift_footprint(self) -> None:
        if isinstance(self.binding.layers[0], probound.conv1d.Conv0d):
            self.skipTest("Can't shift footprint for Conv0d")

        n_iter = 3
        for layer in self.binding.layers:
            if not isinstance(layer, probound.conv1d.Conv1d):
                continue
            self.binding.check_length_consistency()
            self.check_nans(self.binding(self.count_table.seqs))

            for _ in range(n_iter):
                layer.layer_spec.update_footprint(left_shift=1)
                self.binding.check_length_consistency()
                self.check_nans(self.binding(self.count_table.seqs))

            layer.layer_spec.update_footprint(left_shift=-n_iter)
            self.binding.check_length_consistency()
            self.check_nans(self.binding(self.count_table.seqs))

            for _ in range(n_iter):
                layer.layer_spec.update_footprint(right_shift=1)
                self.binding.check_length_consistency()
                self.check_nans(self.binding(self.count_table.seqs))

            layer.layer_spec.update_footprint(right_shift=-n_iter)
            self.binding.check_length_consistency()
            self.check_nans(self.binding(self.count_table.seqs))

    def test_increase_flank(self) -> None:
        n_iter = 5
        for i in range(1, n_iter + 1):
            self.count_table.set_flank_length(left=i, right=i - 1)
            self.binding.update_read_length(left_shift=1)
            self.binding.check_length_consistency()
            self.check_nans(self.binding(self.count_table.seqs))

            self.count_table.set_flank_length(left=i, right=i)
            self.binding.update_read_length(right_shift=1)
            self.binding.check_length_consistency()
            self.check_nans(self.binding(self.count_table.seqs))

        self.count_table.set_flank_length(0, 0)
        self.binding.update_read_length(
            left_shift=-n_iter, right_shift=-n_iter
        )
        self.binding.check_length_consistency()
        self.check_nans(self.binding(self.count_table.seqs))


class TestBindingMode_4di1_bin2(TestBindingMode_0):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.binding = probound.binding.BindingMode(
            [
                probound.conv1d.Conv1d.from_psam(
                    probound.psam.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=4,
                        interaction_distance=1,
                        information_threshold=0.0,
                    ),
                    self.count_table,
                    train_omega=True,
                    train_theta=True,
                    bias_bin=2,
                    one_hot=False,
                )
            ]
        )
        initialize_binding(self.binding)


class TestBindingMode_3di1_3di1_bin2(TestBindingMode_0):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        layer1 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                out_channels=2,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
        )
        layer2 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                kernel_size=3,
                in_channels=2,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            layer1,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
        )
        self.binding = probound.binding.BindingMode([layer1, layer2])
        initialize_binding(self.binding)


class TestBindingMode_3di1_mp2floor(TestBindingMode_0):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        layer1 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                out_channels=2,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
        )
        layer2 = probound.layers.MaxPool1d.from_spec(
            probound.layers.MaxPool1dSpec(
                in_channels=layer1.out_channels, kernel_size=2, ceil_mode=False
            ),
            layer1,
        )
        self.binding = probound.binding.BindingMode([layer1, layer2])
        initialize_binding(self.binding)


class TestBindingMode_3di1_mp2ceil(TestBindingMode_0):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        layer1 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                out_channels=2,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
        )
        layer2 = probound.layers.MaxPool1d.from_spec(
            probound.layers.MaxPool1dSpec(
                in_channels=layer1.out_channels, kernel_size=2, ceil_mode=True
            ),
            layer1,
        )
        self.binding = probound.binding.BindingMode([layer1, layer2])
        initialize_binding(self.binding)


class TestBindingMode_3di1_mp2floor_3di1_bin2(TestBindingMode_0):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        layer1 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                out_channels=2,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
        )
        layer2 = probound.layers.MaxPool1d.from_spec(
            probound.layers.MaxPool1dSpec(
                in_channels=layer1.out_channels, kernel_size=2, ceil_mode=False
            ),
            layer1,
        )
        layer3 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                kernel_size=3,
                in_channels=2,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            layer2,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
        )
        self.binding = probound.binding.BindingMode([layer1, layer2, layer3])
        initialize_binding(self.binding)


class TestBindingMode_3di1_mp2ceil_3di1_bin2(TestBindingMode_0):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        layer1 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                out_channels=2,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            self.count_table,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
        )
        layer2 = probound.layers.MaxPool1d.from_spec(
            probound.layers.MaxPool1dSpec(
                in_channels=layer1.out_channels, kernel_size=2, ceil_mode=True
            ),
            layer1,
        )
        layer3 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                kernel_size=3,
                in_channels=2,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            layer2,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
        )
        self.binding = probound.binding.BindingMode([layer1, layer2, layer3])
        initialize_binding(self.binding)


class TestBindingMode_4di1_bin2_norev(TestBindingMode_0):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.binding = probound.binding.BindingMode(
            [
                probound.conv1d.Conv1d.from_psam(
                    probound.psam.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=4,
                        interaction_distance=1,
                        score_reverse=False,
                        information_threshold=0.0,
                    ),
                    self.count_table,
                    train_omega=True,
                    train_theta=True,
                    bias_bin=2,
                    one_hot=False,
                )
            ]
        )
        initialize_binding(self.binding)


class TestBindingMode_3di1_3di1_bin2_norev(TestBindingMode_0):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        layer1 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                out_channels=2,
                interaction_distance=1,
                score_reverse=False,
                information_threshold=0.0,
            ),
            self.count_table,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
        )
        layer2 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                kernel_size=3,
                in_channels=2,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            layer1,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
        )
        self.binding = probound.binding.BindingMode([layer1, layer2])
        initialize_binding(self.binding)


class TestBindingMode_3di1_mp2floor_3di1_bin2_norev(TestBindingMode_0):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        layer1 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                out_channels=2,
                interaction_distance=1,
                score_reverse=False,
                information_threshold=0.0,
            ),
            self.count_table,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
        )
        layer2 = probound.layers.MaxPool1d.from_spec(
            probound.layers.MaxPool1dSpec(
                in_channels=layer1.out_channels, kernel_size=2, ceil_mode=False
            ),
            layer1,
        )
        layer3 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                kernel_size=3,
                in_channels=2,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            layer2,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
        )
        self.binding = probound.binding.BindingMode([layer1, layer2, layer3])
        initialize_binding(self.binding)


class TestBindingMode_3di1_mp2ceil_3di1_bin2_norev(TestBindingMode_0):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        layer1 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                out_channels=2,
                interaction_distance=1,
                score_reverse=False,
                information_threshold=0.0,
            ),
            self.count_table,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
        )
        layer2 = probound.layers.MaxPool1d.from_spec(
            probound.layers.MaxPool1dSpec(
                in_channels=layer1.out_channels, kernel_size=2, ceil_mode=True
            ),
            layer1,
        )
        layer3 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                kernel_size=3,
                in_channels=2,
                interaction_distance=1,
                information_threshold=0.0,
            ),
            layer2,
            train_omega=True,
            train_theta=True,
            bias_bin=2,
            one_hot=True,
        )
        self.binding = probound.binding.BindingMode([layer1, layer2, layer3])
        initialize_binding(self.binding)


class TestBindingMode_3_3__3_2_sharedPSAM(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.psam = probound.psam.PSAM(
            alphabet=self.count_table.alphabet,
            kernel_size=3,
            information_threshold=0.0,
        )
        bmd0_l1 = probound.conv1d.Conv1d.from_psam(self.psam, self.count_table)
        bmd0_l2 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                kernel_size=3, in_channels=2, information_threshold=0.0
            ),
            bmd0_l1,
        )
        bmd1_l1 = probound.conv1d.Conv1d.from_psam(self.psam, self.count_table)
        bmd1_l2 = probound.conv1d.Conv1d.from_psam(
            probound.psam.PSAM(
                kernel_size=4, in_channels=2, information_threshold=0.0
            ),
            bmd0_l1,
        )
        self.binding_modes = [
            probound.binding.BindingMode([bmd0_l1, bmd0_l2]),
            probound.binding.BindingMode([bmd1_l1, bmd1_l2]),
        ]
        for bmd in self.binding_modes:
            initialize_binding(bmd)

    def test_shift_shared_footprint(self) -> None:
        for bmd in self.binding_modes:
            bmd.check_length_consistency()
        self.psam.update_footprint(left_shift=1)
        for bmd in self.binding_modes:
            bmd.check_length_consistency()
        self.psam.update_footprint(right_shift=1)
        for bmd in self.binding_modes:
            bmd.check_length_consistency()
        self.psam.update_footprint(left_shift=-1, right_shift=-1)
        for bmd in self.binding_modes:
            bmd.check_length_consistency()


class TestBindingMode_3_3_sharedPSAM(TestBindingMode_3_3__3_2_sharedPSAM):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.psam = probound.psam.PSAM(
            alphabet=self.count_table.alphabet,
            kernel_size=3,
            out_channels=4,
            information_threshold=0.0,
        )
        bmd0_l1 = probound.conv1d.Conv1d.from_psam(self.psam, self.count_table)
        bmd0_l2 = probound.conv1d.Conv1d.from_psam(self.psam, bmd0_l1)
        self.binding_modes = [probound.binding.BindingMode([bmd0_l1, bmd0_l2])]
        for bmd in self.binding_modes:
            initialize_binding(bmd)


class TestBindingMode_3_3_flip_sharedPSAM(TestBindingMode_3_3__3_2_sharedPSAM):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.psam = probound.psam.PSAM(
            alphabet=self.count_table.alphabet,
            kernel_size=3,
            out_channels=4,
            information_threshold=0.0,
        )
        bmd0_l2 = probound.conv1d.Conv1d(
            psam=self.psam,
            input_shape=self.count_table.seqs.shape[-1] - 2,
            min_input_length=self.count_table.min_read_length - 2,
            max_input_length=self.count_table.max_read_length - 2,
        )
        bmd0_l1 = probound.conv1d.Conv1d.from_psam(self.psam, self.count_table)
        self.binding_modes = [probound.binding.BindingMode([bmd0_l1, bmd0_l2])]
        for bmd in self.binding_modes:
            initialize_binding(bmd)


if __name__ == "__main__":
    unittest.main()
