# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
import unittest

import torch
from typing_extensions import override

import pyprobound

from . import make_count_table
from .test_layers import BaseTestCases


class TestRollLeft(BaseTestCases.BaseTestLayer):
    layer: pyprobound.layers.Roll

    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.layer = pyprobound.layers.Roll.from_spec(
            pyprobound.layers.RollSpec(
                alphabet=self.count_table.alphabet,
                direction="left",
                max_length=5,
            ),
            self.count_table,
        )

    def test_padding(self) -> None:
        seqs = []
        direction = self.layer.layer_spec.direction
        max_length = self.layer.layer_spec.max_length
        rolls = self.count_table.input_shape - self.layer.lengths(
            self.count_table.seqs
        )
        for seq, roll in zip(self.count_table.seqs, rolls):
            if direction != "left":
                if direction == "center":
                    roll //= 2
                seq = seq.roll(shifts=roll.item(), dims=-1)
            if max_length is not None:
                if direction == "left":
                    seq = seq[..., :max_length]
                elif direction == "right":
                    seq = seq[..., -max_length:]
            seqs.append(seq)

        self.assertTrue(
            torch.equal(torch.stack(seqs), self.layer(self.count_table.seqs)),
            "Incorrect implementation of vectorized roll",
        )


class TestRollRight(TestRollLeft):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.layer = pyprobound.layers.Roll.from_spec(
            pyprobound.layers.RollSpec(
                alphabet=self.count_table.alphabet,
                direction="right",
                max_length=5,
            ),
            self.count_table,
        )


class TestRollCenter(TestRollLeft):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()
        self.layer = pyprobound.layers.Roll.from_spec(
            pyprobound.layers.RollSpec(
                alphabet=self.count_table.alphabet, direction="center"
            ),
            self.count_table,
        )

    @override
    def test_update_limit(self) -> None:
        self.skipTest(
            "Center padding yields different outputs"
            " between inputs with odd or even number of finite elements"
        )


if __name__ == "__main__":
    unittest.main()
