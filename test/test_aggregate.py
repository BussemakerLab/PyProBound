# pylint: disable=missing-class-docstring, missing-module-docstring
import math
import unittest

from typing_extensions import override

import probound.aggregate
import probound.alphabet
import probound.binding
import probound.conv1d
import probound.table


class TestAggregate(unittest.TestCase):
    @override
    def setUp(self) -> None:
        alphabet = probound.alphabet.DNA()

        bmd_0 = probound.binding.BindingMode(
            [
                probound.conv1d.Conv0d(
                    nonspecific=probound.psam.NonSpecific(alphabet=alphabet),
                    input_shape=20,
                    min_input_length=20,
                    max_input_length=20,
                )
            ]
        )
        bmd_1 = probound.binding.BindingMode(
            [
                probound.conv1d.Conv1d(
                    psam=probound.psam.PSAM(kernel_size=1, alphabet=alphabet),
                    input_shape=20,
                    min_input_length=20,
                    max_input_length=20,
                )
            ]
        )
        bmd_2 = probound.binding.BindingMode(
            [
                probound.conv1d.Conv1d(
                    psam=probound.psam.PSAM(kernel_size=1, alphabet=alphabet),
                    input_shape=20,
                    min_input_length=20,
                    max_input_length=20,
                )
            ]
        )

        self.aggregate = probound.aggregate.Aggregate(
            (
                probound.aggregate.Contribution(bmd)
                for bmd in (bmd_0, bmd_1, bmd_2)
            ),
            target_concentration=100,
        )

    def test_activity_heuristic(self) -> None:
        frac = 0.05

        self.assertEqual(
            self.aggregate.expected_log_aggregate(),
            float("-inf"),
            "Initial expected_log_aggregate not -inf",
        )

        for ctrb in self.aggregate.contributions:
            self.aggregate.alpha_heuristic(ctrb, frac)

            self.assertAlmostEqual(
                self.aggregate.expected_log_aggregate(),
                self.aggregate.log_target_concentration.item(),
                places=6,
                msg=f"Training {ctrb} did not keep expected_log_aggregate=0.0",
            )

            # pylint: disable-next=protected-access
            if len(list(self.aggregate._contributing())) > 1:
                self.assertAlmostEqual(
                    ctrb.expected_log_contribution(),
                    math.log(frac)
                    + self.aggregate.expected_log_aggregate()
                    - self.aggregate.log_target_concentration.item(),
                    places=6,
                    msg=f"{ctrb} does not contribute {frac:.2%} to aggregate",
                )


if __name__ == "__main__":
    unittest.main()
