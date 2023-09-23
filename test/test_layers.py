# pylint: disable=invalid-name, missing-class-docstring, missing-module-docstring, protected-access
import unittest

import torch
import torch.nn.functional as F
from typing_extensions import override

import probound.layers
import probound.table

from . import make_count_table


class BaseTestCases:
    class BaseTestLayer(unittest.TestCase):
        layer: probound.layers.Layer
        count_table: probound.table.CountTable

        def test_out_len_shape(self) -> None:
            num_seqs = len(self.count_table)
            num_pos = self.count_table.input_shape
            self.assertEqual(
                self.layer(self.count_table.seqs).shape,
                (
                    num_seqs,
                    self.layer.out_channels,
                    self.layer.out_len(num_pos, mode="shape"),
                ),
                "incorrect out shape",
            )

        def test_out_len_finite(self) -> None:
            self.layer._min_input_length = torch.tensor(float("-inf"))
            self.layer._max_input_length = torch.tensor(float("inf"))

            # get padding length
            pad_len = self.layer.in_len(1, "max")
            if pad_len is None:
                pad_len = self.layer.in_len(1, "min")

            # get out_len output
            lengths = self.layer.lengths(self.count_table.seqs)
            out_len_min = self.layer.out_len(lengths, mode="min")
            out_len_max = self.layer.out_len(lengths, mode="max")

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
                output = self.layer(seqs).isfinite().sum(-1)
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

        def test_in_len(self) -> None:
            self.layer._min_input_length = torch.tensor(float("-inf"))
            self.layer._max_input_length = torch.tensor(float("inf"))
            for out_len in range(2, 5):
                min_len = self.layer.in_len(out_len, "min")
                max_len = self.layer.in_len(out_len, "max")
                if max_len is None:
                    self.skipTest("Can't test in_len if output is undefined")

                self.assertEqual(
                    self.layer(
                        self.count_table.seqs[..., : min_len - 1]
                    ).shape[-1],
                    out_len - 1,
                    "size('min') output incorrect",
                )
                self.assertEqual(
                    self.layer(self.count_table.seqs[..., :min_len]).shape[-1],
                    out_len,
                    "size('min') output incorrect",
                )
                self.assertEqual(
                    self.layer(self.count_table.seqs[..., :max_len]).shape[-1],
                    out_len,
                    "size('max') output incorrect",
                )
                self.assertEqual(
                    self.layer(
                        self.count_table.seqs[..., : max_len + 1]
                    ).shape[-1],
                    out_len + 1,
                    "size('max') output incorrect",
                )

        def test_update_limit(self) -> None:
            # baseline
            prev_score = self.layer(self.count_table.seqs)
            prev_shape = self.count_table.seqs.shape
            self.assertEqual(
                self.layer.lengths(self.count_table.seqs).min().item(),
                self.layer.min_input_length,
                "incorrect min_input_length",
            )
            self.assertEqual(
                self.layer.lengths(self.count_table.seqs).max().item(),
                self.layer.max_input_length,
                "incorrect max_input_length",
            )

            # insert a sequence shorter than the shortest in the count table
            insert = self.count_table.seqs[-1:].clone()
            pad_val: float = self.count_table.alphabet.neginf_pad
            if self.count_table.seqs.ndim == 3:
                pad_val = float("-inf")
            insert[..., self.count_table.min_variable_length - 1 :] = pad_val
            self.count_table.seqs = torch.cat([self.count_table.seqs, insert])

            self.layer.update_input_length(min_len_shift=-1)
            self.assertEqual(
                self.layer.lengths(self.count_table.seqs).min().item(),
                self.layer.min_input_length,
                "incorrect min_input_length after increasing min input length",
            )
            self.assertEqual(
                self.layer.lengths(self.count_table.seqs).max().item(),
                self.layer.max_input_length,
                "incorrect max_input_length after increasing min input length",
            )

            curr_score = self.layer(self.count_table.seqs)
            self.assertTrue(
                torch.allclose(prev_score, curr_score[:-1]),
                "incorrect output after decreasing min input length",
            )

            # reset count table
            self.count_table.seqs = self.count_table.seqs[:-1]
            self.layer.update_input_length(min_len_shift=1)
            self.assertEqual(
                self.layer.lengths(self.count_table.seqs).min().item(),
                self.layer.min_input_length,
                "incorrect min_input_length after resetting min input length",
            )
            self.assertEqual(
                self.layer.lengths(self.count_table.seqs).max().item(),
                self.layer.max_input_length,
                "incorrect max_input_length after resetting min input length",
            )
            curr_score = self.layer(self.count_table.seqs)
            self.assertTrue(
                torch.allclose(prev_score, curr_score),
                "incorrect output after resetting min input length",
            )

            # insert a sequence longer than the longest in the count table
            self.count_table.seqs = F.pad(
                self.count_table.seqs, pad=(0, 1), value=pad_val
            )
            insert = self.count_table.seqs[-1:].clone()
            torch.nn.init.zeros_(insert)
            self.count_table.seqs = torch.cat([self.count_table.seqs, insert])

            self.layer.update_input_length(
                right_shift=1,
                max_len_shift=1,
                new_min_len=self.layer.min_input_length,
                new_max_len=self.layer.max_input_length + 1,
            )
            self.assertEqual(
                self.layer.lengths(self.count_table.seqs).min().item(),
                self.layer.min_input_length,
                "incorrect min_input_length after increasing max input length",
            )
            self.assertEqual(
                self.layer.lengths(self.count_table.seqs).max().item(),
                self.layer.max_input_length,
                "incorrect max_input_length after increasing max input length",
            )

            curr_score = self.layer(self.count_table.seqs)
            if prev_score.shape[-1] != curr_score.shape[-1]:
                curr_score = curr_score[
                    ..., : prev_score.shape[-1] - curr_score.shape[-1]
                ]
            self.assertTrue(
                torch.allclose(prev_score, curr_score[:-1]),
                "incorrect output after increasing max input length",
            )

            # reset count table
            self.count_table.seqs = self.count_table.seqs[:-1, ..., :-1]
            self.assertEqual(
                self.count_table.seqs.shape,
                prev_shape,
                "incorrect seqs shape after resetting max input length",
            )
            self.layer.update_input_length(
                right_shift=-1,
                max_len_shift=-1,
                new_min_len=self.layer.min_input_length,
                new_max_len=self.layer.max_input_length - 1,
            )
            self.assertEqual(
                self.layer.lengths(self.count_table.seqs).min().item(),
                self.layer.min_input_length,
                "incorrect min_input_length after resetting max input length",
            )
            self.assertEqual(
                self.layer.lengths(self.count_table.seqs).max().item(),
                self.layer.max_input_length,
                "incorrect max_input_length after resetting max input length",
            )
            curr_score = self.layer(self.count_table.seqs)
            self.assertTrue(
                torch.allclose(prev_score, curr_score),
                "incorrect output after resetting max input length",
            )


class TestMaxPool1d_3ceil(BaseTestCases.BaseTestLayer):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.count_table.seqs = self.count_table.alphabet.embedding(
            self.count_table.seqs
        ).transpose(1, 2)
        self.layer = probound.layers.MaxPool1d.from_spec(
            probound.layers.MaxPool1dSpec(
                in_channels=len(self.count_table.alphabet.alphabet),
                kernel_size=3,
                ceil_mode=True,
            ),
            self.count_table,
        )


class TestMaxPool1d_3floor(BaseTestCases.BaseTestLayer):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.count_table.seqs = self.count_table.alphabet.embedding(
            self.count_table.seqs
        ).transpose(1, 2)
        self.layer = probound.layers.MaxPool1d.from_spec(
            probound.layers.MaxPool1dSpec(
                in_channels=len(self.count_table.alphabet.alphabet),
                kernel_size=3,
                ceil_mode=False,
            ),
            self.count_table,
        )


class TestMaxPool1d_2ceil(BaseTestCases.BaseTestLayer):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(max_input_length=21)
        self.count_table.set_flank_length(0, 0)
        self.count_table.seqs = self.count_table.alphabet.embedding(
            self.count_table.seqs
        ).transpose(1, 2)
        self.layer = probound.layers.MaxPool1d.from_spec(
            probound.layers.MaxPool1dSpec(
                in_channels=len(self.count_table.alphabet.alphabet),
                kernel_size=2,
                ceil_mode=True,
            ),
            self.count_table,
        )


class TestMaxPool1d_2floor(BaseTestCases.BaseTestLayer):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table(max_input_length=21)
        self.count_table.set_flank_length(0, 0)
        self.count_table.seqs = self.count_table.alphabet.embedding(
            self.count_table.seqs
        ).transpose(1, 2)
        self.layer = probound.layers.MaxPool1d.from_spec(
            probound.layers.MaxPool1dSpec(
                in_channels=len(self.count_table.alphabet.alphabet),
                kernel_size=2,
                ceil_mode=False,
            ),
            self.count_table,
        )


if __name__ == "__main__":
    unittest.main()
