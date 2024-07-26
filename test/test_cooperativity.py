# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, missing-module-docstring
import unittest

import torch
from torch import Tensor
from typing_extensions import override

import pyprobound

from . import make_count_table
from .test_binding import initialize_binding


def initialize_cooperativity(
    binding_cooperativity: pyprobound.Cooperativity,
) -> None:
    torch.nn.init.normal_(binding_cooperativity.spacing.log_spacing)
    if binding_cooperativity.train_posbias:
        for diagonal in binding_cooperativity.log_posbias:
            torch.nn.init.normal_(diagonal, std=10)
    else:
        binding_cooperativity.log_posbias.requires_grad_(False)
    for bmd in (binding_cooperativity.mode_a, binding_cooperativity.mode_b):
        initialize_binding(bmd)


class TestCooperativity_5__3(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(), binding_mode_b.key(), normalize=False
            ),
            binding_mode_a,
            binding_mode_b,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()

    def test_score_mode(self) -> None:
        spacing = self.binding_cooperativity.spacing
        matrix = self.binding_cooperativity.get_log_spacing_matrix()[-1, -1]

        # Test score_reverse and score_same
        if spacing.n_strands > 1:
            if not spacing.score_reverse:
                self.assertTrue(
                    torch.all(torch.isneginf(matrix[0, 1]))
                    and torch.all(torch.isneginf(matrix[1, 0])),
                    "Expected opposite-strand spacing parameters to be -inf",
                )
            if not spacing.score_same:
                self.assertTrue(
                    torch.all(torch.isneginf(matrix[0, 0]))
                    and torch.all(torch.isneginf(matrix[0, 0])),
                    "Expected same-strand spacing parameters to be -inf",
                )

        # Test score_mode
        if spacing.score_mode == "positive":
            offsets = range(0, matrix.shape[-2], -1)
        elif spacing.score_mode == "negative":
            offsets = range(1, matrix.shape[-1])
        else:
            offsets = range(0, 0)
        for offset in offsets:
            self.assertTrue(
                torch.all(
                    torch.isneginf(torch.diagonal(matrix, offset, -2, -1))
                ),
                f"Expected {spacing.score_mode} spacing parameters to be -inf",
            )

    def test_spacing_gauge(self) -> None:
        spacing = self.binding_cooperativity.spacing
        zero_gauge: Tensor | None = None
        for i in range(1, spacing.max_num_windows + 1):
            for j in range(1, spacing.max_num_windows + 1):
                if zero_gauge is None:
                    zero_gauge = spacing.get_log_spacing_matrix(i, j)[
                        0, 0, 0, 0
                    ]
                self.assertEqual(
                    zero_gauge,
                    spacing.get_log_spacing_matrix(i, j)[0, 0, 0, 0],
                    "Spacing main diagonal inconsistent with changing windows",
                )

    def test_spacing(self) -> None:
        for param in self.binding_cooperativity.log_posbias:
            torch.nn.init.zeros_(param)
        # pylint: disable-next=protected-access
        self.binding_cooperativity._length_specific_bias = False

        mode_a = self.binding_cooperativity.mode_a
        mode_b = self.binding_cooperativity.mode_b
        n_windows_a = mode_a.out_len(self.count_table.seqs.shape[-1])
        n_windows_b = mode_b.out_len(self.count_table.seqs.shape[-1])
        log_spacing = self.binding_cooperativity.get_log_spacing()

        with (
            unittest.mock.patch.object(mode_a, "score_windows") as mock_a,
            unittest.mock.patch.object(mode_b, "score_windows") as mock_b,
        ):
            for offset in range(-n_windows_a + 1, n_windows_b):
                start_a = max(0, -offset)
                end_a = min(n_windows_a, n_windows_b - offset)
                start_b = max(0, offset)
                end_b = min(n_windows_b, n_windows_a + offset)
                n_seqs = end_a - start_a
                assert n_seqs == (end_b - start_b)

                for strand_a in range(mode_a.out_channels):
                    for strand_b in range(mode_b.out_channels):
                        log_spacing_value = log_spacing[
                            strand_a, strand_b, offset + n_windows_a - 1
                        ]
                        if torch.isneginf(log_spacing_value):
                            continue

                        # Fill windows so each row contains
                        windows_a = torch.full(
                            (n_seqs, mode_a.out_channels, n_windows_a),
                            float("-inf"),
                        )
                        windows_a[
                            torch.arange(n_seqs),
                            strand_a,
                            torch.arange(start_a, end_a),
                        ] = 0
                        mock_a.return_value = windows_a

                        windows_b = torch.full(
                            (n_seqs, mode_b.out_channels, n_windows_b),
                            float("-inf"),
                        )
                        windows_b[
                            torch.arange(n_seqs),
                            strand_b,
                            torch.arange(start_b, end_b),
                        ] = 0
                        mock_b.return_value = windows_b

                        self.assertTrue(
                            torch.all(
                                self.binding_cooperativity.score_windows(
                                    self.count_table.seqs[:n_seqs]
                                )
                                .isfinite()
                                .sum((-1, -2, -3, -4))
                                == 1
                            ),
                            f"Failed to isolate output for offset {offset} and"
                            f" strands {(strand_a, strand_b)}",
                        )

                        self.assertTrue(
                            torch.all(
                                self.binding_cooperativity(
                                    self.count_table.seqs[:n_seqs]
                                )
                                == log_spacing_value
                            ),
                            f"Isolated offset {offset} does not match"
                            " log_spacing value",
                        )

    def test_overlap(self) -> None:
        spacing = self.binding_cooperativity.spacing
        mode_a = self.binding_cooperativity.mode_a
        mode_b = self.binding_cooperativity.mode_b
        if (mode_a.in_len(1, mode="min") != mode_a.in_len(1, mode="max")) or (
            mode_b.in_len(1, mode="min") != mode_b.in_len(1, mode="max")
        ):
            self.skipTest("Can't test overlap if receptive field is variable")

        for overlap in range(3):
            # Set max_overlap = -max_spacing to create a single coop param
            spacing.max_overlap = overlap
            spacing.max_spacing = -overlap
            self.assertTrue(
                torch.all(
                    torch.isfinite(
                        self.binding_cooperativity.get_log_spacing()
                    ).sum(dim=-1)
                    <= 2
                ),
                f"Setting max_overlap={overlap} & max_spacing=-{overlap} did"
                " not result in a single spacing parameter per orientation",
            )

            # Create diagonal neginf sequences to identify binding footprint
            seqs = torch.zeros(
                (self.count_table.seqs.shape[-1],) * 2,
                dtype=self.count_table.seqs.dtype,
            )
            diag_seqs = seqs.clone().fill_diagonal_(
                self.count_table.alphabet.neginf_pad
            )

            # Get cooperativity output
            with unittest.mock.patch.object(
                mode_a,
                "score_windows",
                return_value=mode_a.score_windows(diag_seqs),
            ):
                coop_out_a = self.binding_cooperativity.score_windows(seqs)
            with unittest.mock.patch.object(
                mode_b,
                "score_windows",
                return_value=mode_b.score_windows(diag_seqs),
            ):
                coop_out_b = self.binding_cooperativity.score_windows(seqs)

            # Check binding footprints at all relative offsets of given overlap
            for idx in (
                self.binding_cooperativity.get_log_spacing_matrix()
                .isfinite()[-1, -1]
                .nonzero()
            ):
                # Can't use * with Python 3.10
                footprint_a = coop_out_a[
                    :, idx[0], idx[1], idx[2], idx[3]
                ].isneginf()
                footprint_b = coop_out_b[
                    :, idx[0], idx[1], idx[2], idx[3]
                ].isneginf()
                self.assertEqual(
                    torch.sum(footprint_a & footprint_b),
                    overlap,
                    f"There is not an overlap of {overlap} at index {idx}",
                )

    def check_nans(self, tensor: Tensor) -> None:
        self.assertFalse(
            torch.any(torch.isnan(tensor)), "output contains NaNs"
        )

    def test_coop_v_spacing(self) -> None:
        for param in self.binding_cooperativity.log_posbias:
            torch.nn.init.zeros_(param)
        self.binding_cooperativity.log_posbias.requires_grad_(True)
        coop_matrix = self.binding_cooperativity.get_log_spacing_matrix()
        spacing_matrix = (
            self.binding_cooperativity.spacing.get_log_spacing_matrix(
                self.binding_cooperativity.n_windows_a,
                self.binding_cooperativity.n_windows_b,
            )
        )
        while coop_matrix.ndim > spacing_matrix.ndim:
            coop_matrix = coop_matrix[-1]
        self.assertTrue(
            torch.equal(coop_matrix, spacing_matrix),
            "spacing and cooperativity log_spacing_matrices do not match",
        )

    def test_matrix_v_diag(self) -> None:
        for param in self.binding_cooperativity.log_posbias:
            torch.nn.init.zeros_(param)
        self.binding_cooperativity.log_posbias.requires_grad_(False)
        diag_out = self.binding_cooperativity(self.count_table.seqs)
        self.binding_cooperativity.log_posbias.requires_grad_(True)
        matrix_out = self.binding_cooperativity(self.count_table.seqs)
        self.check_nans(diag_out)
        self.check_nans(matrix_out)
        self.assertTrue(
            torch.allclose(diag_out, matrix_out, atol=1e-6),
            "diagonal and matrix outputs do not match",
        )

    def test_windows_v_forward(self) -> None:
        dense_out = self.binding_cooperativity(self.count_table.seqs)
        window_out = (
            torch.exp(self.binding_cooperativity.log_hill)
            * self.binding_cooperativity.score_windows(self.count_table.seqs)
        ).logsumexp((-1, -2, -3, -4))
        self.check_nans(dense_out)
        self.check_nans(window_out)
        self.assertTrue(
            torch.allclose(dense_out, window_out, atol=1e-6),
            "dense and window outputs do not match",
        )

    def test_shift_footprint(self) -> None:
        n_iter = 3
        for bmd in self.binding_cooperativity.components():
            for layer in bmd.layers:
                if not isinstance(layer, pyprobound.layers.Conv1d):
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
            all(isinstance(i, pyprobound.layers.Conv1d) for i in bmd.layers)
            for bmd in self.binding_cooperativity.components()
        ):
            self.skipTest("Can't test shift spacing if any non-Conv1d layers")
        if self.binding_cooperativity.bias_mode == "reverse":
            self.skipTest("Can't test shift spacing if reverse-mode posbias")
        if self.binding_cooperativity.normalize:
            self.skipTest("Can't increase flank with normalized posbias")

        # store and strip out length information
        info: list[dict[str, torch.nn.Parameter]] = []
        for bmd in self.binding_cooperativity.components():
            for layer in bmd.layers:
                if not isinstance(layer, pyprobound.layers.Conv1d):
                    info.append({})
                    continue
                info.append({"prev_posbias": layer.log_posbias})
                layer.log_posbias.requires_grad_()
                layer.log_posbias.zero_()

        # establish baseline
        prev_score = self.binding_cooperativity(self.count_table.seqs)
        self.check_nans(prev_score)
        shift = self.binding_cooperativity.bias_bin

        # shift left
        self.count_table.set_flank_length(left=shift, right=0)
        for bmd in self.binding_cooperativity.components():
            bmd.update_read_length(left_shift=shift)
        return_matrix = self.binding_cooperativity.get_log_spacing_matrix()
        with unittest.mock.patch.object(
            self.binding_cooperativity, "get_log_spacing_matrix"
        ) as log_spacing_matrix:
            return_matrix[..., 0] = float("-inf")
            return_matrix[..., 0, :] = float("-inf")
            log_spacing_matrix.return_value = return_matrix
            curr_score = (
                torch.exp(self.binding_cooperativity.log_hill)
                * self.binding_cooperativity.score_windows(
                    self.count_table.seqs
                )
            ).logsumexp((-1, -2, -3, -4))
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
            "incorrect output after decreasing left flank",
        )


class TestCooperativity_3__5(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(), binding_mode_b.key(), normalize=False
            ),
            binding_mode_a,
            binding_mode_b,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__norev(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                score_reverse=False,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                score_reverse=False,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(), binding_mode_b.key(), normalize=False
            ),
            binding_mode_a,
            binding_mode_b,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__posbias(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(), binding_mode_b.key(), normalize=False
            ),
            binding_mode_a,
            binding_mode_b,
            train_posbias=True,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__posbias_bin5(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(), binding_mode_b.key(), normalize=False
            ),
            binding_mode_a,
            binding_mode_b,
            train_posbias=True,
            bias_bin=5,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__posbiasnorm(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(), binding_mode_b.key(), normalize=False
            ),
            binding_mode_a,
            binding_mode_b,
            train_posbias=True,
            normalize=True,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__hill(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(), binding_mode_b.key(), normalize=False
            ),
            binding_mode_a,
            binding_mode_b,
            train_hill=True,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__posbias_norev(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                score_reverse=False,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                score_reverse=False,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(), binding_mode_b.key(), normalize=False
            ),
            binding_mode_a,
            binding_mode_b,
            train_posbias=True,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__posbias_over2_spacing5(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(),
                binding_mode_b.key(),
                max_overlap=2,
                max_spacing=5,
                normalize=False,
            ),
            binding_mode_a,
            binding_mode_b,
            train_posbias=True,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__no_length_posbias(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(), binding_mode_b.key(), normalize=False
            ),
            binding_mode_a,
            binding_mode_b,
            train_posbias=True,
            normalize=False,
            length_specific_bias=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__same_posbias(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(), binding_mode_b.key(), normalize=False
            ),
            binding_mode_a,
            binding_mode_b,
            train_posbias=True,
            normalize=False,
            bias_mode="same",
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__reverse_posbias(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(), binding_mode_b.key(), normalize=False
            ),
            binding_mode_a,
            binding_mode_b,
            train_posbias=True,
            normalize=False,
            bias_mode="reverse",
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__scoresame(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(),
                binding_mode_b.key(),
                normalize=False,
                score_reverse=False,
            ),
            binding_mode_a,
            binding_mode_b,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__scorerev(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(),
                binding_mode_b.key(),
                normalize=False,
                score_same=False,
            ),
            binding_mode_a,
            binding_mode_b,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__scorepos(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(),
                binding_mode_b.key(),
                normalize=False,
                score_mode="positive",
            ),
            binding_mode_a,
            binding_mode_b,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3__scoreneg(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(),
                binding_mode_b.key(),
                normalize=False,
                score_mode="negative",
            ),
            binding_mode_a,
            binding_mode_b,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_3_3_bin1__3_3_bin2(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        bmd0_l1 = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            train_posbias=True,
            unfold=True,
            bias_bin=1,
            normalize=False,
        )
        bmd0_l2 = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                kernel_size=3,
                in_channels=2,
                information_threshold=0.0,
                normalize=False,
            ),
            bmd0_l1,
            train_posbias=True,
            unfold=True,
            bias_bin=1,
            one_hot=True,
            normalize=False,
        )
        bmd1_l1 = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            train_posbias=True,
            unfold=True,
            bias_bin=2,
            normalize=False,
        )
        bmd1_l2 = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                kernel_size=3,
                in_channels=2,
                information_threshold=0.0,
                normalize=False,
            ),
            bmd1_l1,
            train_posbias=True,
            unfold=True,
            bias_bin=2,
            one_hot=True,
            normalize=False,
        )
        binding_mode_a = pyprobound.Mode([bmd0_l1, bmd0_l2])
        binding_mode_b = pyprobound.Mode([bmd1_l1, bmd1_l2])
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(), binding_mode_b.key(), normalize=False
            ),
            binding_mode_a,
            binding_mode_b,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_3_mp2floor_3__3_3mp2ceil_3(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        bmd0_l1 = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            train_posbias=True,
            unfold=True,
            normalize=False,
        )
        bmd0_l2 = pyprobound.layers.MaxPool1d.from_spec(
            pyprobound.layers.MaxPool1dSpec(
                in_channels=bmd0_l1.out_channels,
                kernel_size=2,
                ceil_mode=False,
            ),
            bmd0_l1,
        )
        bmd0_l3 = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                kernel_size=3,
                in_channels=2,
                information_threshold=0.0,
                normalize=False,
            ),
            bmd0_l2,
            train_posbias=True,
            unfold=True,
            one_hot=True,
            normalize=False,
        )
        bmd1_l1 = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            train_posbias=True,
            unfold=True,
            normalize=False,
        )
        bmd1_l2 = pyprobound.layers.MaxPool1d.from_spec(
            pyprobound.layers.MaxPool1dSpec(
                in_channels=bmd1_l1.out_channels, kernel_size=2, ceil_mode=True
            ),
            bmd1_l1,
        )
        bmd1_l3 = pyprobound.layers.Conv1d.from_psam(
            pyprobound.layers.PSAM(
                kernel_size=3,
                in_channels=2,
                information_threshold=0.0,
                normalize=False,
            ),
            bmd1_l2,
            train_posbias=True,
            unfold=True,
            one_hot=True,
            normalize=False,
        )
        binding_mode_a = pyprobound.Mode([bmd0_l1, bmd0_l2, bmd0_l3])
        binding_mode_b = pyprobound.Mode([bmd1_l1, bmd1_l2, bmd1_l3])
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(), binding_mode_b.key(), normalize=False
            ),
            binding_mode_a,
            binding_mode_b,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5pad2__3(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(
            min_input_length=20, max_input_length=20
        )
        pad = pyprobound.layers.Pad.from_spec(
            pyprobound.layers.PadSpec(
                self.count_table.alphabet, left=2, right=2
            ),
            self.count_table,
        )
        binding_mode_a = pyprobound.Mode(
            [
                pad,
                pyprobound.layers.Conv1d.from_psam(
                    pyprobound.layers.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=5,
                        information_threshold=0.0,
                        normalize=False,
                    ),
                    pad,
                    normalize=False,
                ),
            ]
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(),
                binding_mode_b.key(),
                normalize=False,
                ignore_pad=True,
            ),
            binding_mode_a,
            binding_mode_b,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3padm2(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(
            min_input_length=20, max_input_length=20
        )
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        pad = pyprobound.layers.Pad.from_spec(
            pyprobound.layers.PadSpec(
                self.count_table.alphabet, left=-2, right=-2
            ),
            self.count_table,
        )
        binding_mode_b = pyprobound.Mode(
            [
                pad,
                pyprobound.layers.Conv1d.from_psam(
                    pyprobound.layers.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=3,
                        information_threshold=0.0,
                        normalize=False,
                    ),
                    pad,
                    normalize=False,
                ),
            ]
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(),
                binding_mode_b.key(),
                normalize=False,
                ignore_pad=True,
            ),
            binding_mode_a,
            binding_mode_b,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5__3padm3left(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(
            min_input_length=20, max_input_length=20
        )
        binding_mode_a = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=5,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        pad = pyprobound.layers.Pad.from_spec(
            pyprobound.layers.PadSpec(self.count_table.alphabet, left=-3),
            self.count_table,
        )
        binding_mode_b = pyprobound.Mode(
            [
                pad,
                pyprobound.layers.Conv1d.from_psam(
                    pyprobound.layers.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=3,
                        information_threshold=0.0,
                        normalize=False,
                    ),
                    pad,
                    normalize=False,
                ),
            ]
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(),
                binding_mode_b.key(),
                normalize=False,
                ignore_pad=True,
            ),
            binding_mode_a,
            binding_mode_b,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_5pad3right__3(TestCooperativity_5__3):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(
            min_input_length=20, max_input_length=20
        )
        pad = pyprobound.layers.Pad.from_spec(
            pyprobound.layers.PadSpec(self.count_table.alphabet, right=3),
            self.count_table,
        )
        binding_mode_a = pyprobound.Mode(
            [
                pad,
                pyprobound.layers.Conv1d.from_psam(
                    pyprobound.layers.PSAM(
                        alphabet=self.count_table.alphabet,
                        kernel_size=5,
                        information_threshold=0.0,
                        normalize=False,
                    ),
                    pad,
                    normalize=False,
                ),
            ]
        )
        binding_mode_b = pyprobound.Mode.from_psam(
            pyprobound.layers.PSAM(
                alphabet=self.count_table.alphabet,
                kernel_size=3,
                information_threshold=0.0,
                normalize=False,
            ),
            self.count_table,
            normalize=False,
        )
        self.binding_cooperativity = pyprobound.Cooperativity(
            pyprobound.Spacing(
                binding_mode_a.key(),
                binding_mode_b.key(),
                normalize=False,
                ignore_pad=True,
            ),
            binding_mode_a,
            binding_mode_b,
            normalize=False,
        )
        initialize_cooperativity(self.binding_cooperativity)
        self.binding_cooperativity.check_length_consistency()


class TestCooperativity_3_sharedSpacing(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.psam_b = pyprobound.layers.PSAM(
            alphabet=self.count_table.alphabet,
            kernel_size=3,
            information_threshold=0.0,
            normalize=False,
        )
        self.spacing = pyprobound.Spacing([self.psam_b], [self.psam_b])
        self.coops = [
            pyprobound.Cooperativity(
                self.spacing,
                pyprobound.Mode(
                    [
                        pyprobound.layers.Conv1d.from_psam(
                            self.psam_b, self.count_table, normalize=False
                        )
                    ]
                ),
                pyprobound.Mode(
                    [
                        pyprobound.layers.Conv1d.from_psam(
                            self.psam_b, self.count_table, normalize=False
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
        self.psam_b.update_footprint(left_shift=1)
        for coop in self.coops:
            coop.check_length_consistency()
        self.psam_b.update_footprint(right_shift=1)
        for coop in self.coops:
            coop.check_length_consistency()
        self.psam_b.update_footprint(left_shift=-1, right_shift=-1)
        for coop in self.coops:
            coop.check_length_consistency()
        self.psam_b.update_footprint(left_shift=1)
        for coop in self.coops:
            coop.check_length_consistency()
        self.psam_b.update_footprint(right_shift=1)
        for coop in self.coops:
            coop.check_length_consistency()
        self.psam_b.update_footprint(left_shift=-1, right_shift=-1)
        for coop in self.coops:
            coop.check_length_consistency()


class TestCooperativity_3_sharedMode(TestCooperativity_3_sharedSpacing):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.psam_b = pyprobound.layers.PSAM(
            alphabet=self.count_table.alphabet,
            kernel_size=3,
            information_threshold=0.0,
            normalize=False,
        )
        self.spacing = pyprobound.Spacing([self.psam_b], [self.psam_b])
        self.mode = pyprobound.Mode.from_psam(
            self.psam_b, self.count_table, normalize=False
        )
        self.coops = [
            pyprobound.Cooperativity(self.spacing, self.mode, self.mode)
            for _ in range(2)
        ]
        for coop in self.coops:
            initialize_cooperativity(coop)
            coop.check_length_consistency()


if __name__ == "__main__":
    unittest.main()
