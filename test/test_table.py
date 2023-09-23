# pylint: disable=missing-class-docstring, missing-module-docstring
import unittest

import torch
from typing_extensions import override

from . import make_count_table


class TestCountTable(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.count_table = make_count_table()

    def test_flank_length(self) -> None:
        self.count_table.set_flank_length(0, 0)
        var_region = self.count_table.seqs

        # increment left flank
        self.count_table.set_flank_length(1, 0)
        new_flank = self.count_table.alphabet.translate(
            self.count_table.left_flank[-1]
        )
        self.assertTrue(
            torch.all(
                self.count_table.seqs
                == torch.cat(
                    [new_flank.expand(len(self.count_table), -1), var_region],
                    dim=1,
                )
            ),
            "couldn't increment left flank",
        )

        # revert
        self.count_table.set_flank_length(0, 0)
        self.assertTrue(
            torch.all(self.count_table.seqs == var_region),
            "couldn't revert from increment left flank",
        )

        # increment right flank
        self.count_table.set_flank_length(0, 1)
        new_flank = self.count_table.alphabet.translate(
            self.count_table.right_flank[0]
        )
        self.assertTrue(
            torch.all(
                self.count_table.seqs
                == torch.stack(
                    [
                        torch.cat(
                            [
                                var_region[
                                    i, : self.count_table.variable_lengths[i]
                                ],
                                new_flank,
                                var_region[
                                    i, self.count_table.variable_lengths[i] :
                                ],
                            ]
                        )
                        for i in range(len(self.count_table))
                    ]
                )
            ),
            "couldn't increment right flank",
        )

        # revert
        self.count_table.set_flank_length(0, 0)
        self.assertTrue(
            torch.all(self.count_table.seqs == var_region),
            "couldn't revert from increment right flank",
        )


if __name__ == "__main__":
    unittest.main()
