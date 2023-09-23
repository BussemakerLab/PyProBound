# pylint: disable=missing-class-docstring, missing-module-docstring, protected-access
import collections
import unittest

from typing_extensions import override

import probound.base
import probound.binding
import probound.conv1d
import probound.cooperativity
import probound.experiment
import probound.loss
import probound.psam
import probound.rounds

from . import make_count_table


class TestCaching(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.count_tables = [
            make_count_table(n_columns=3),
            make_count_table(n_columns=2),
        ]

        psam_0 = probound.psam.PSAM(1, self.count_tables[0].alphabet)
        psam_1 = probound.psam.PSAM(1, self.count_tables[0].alphabet)
        conv1d_0_expt_0 = probound.conv1d.Conv1d.from_psam(
            psam_0, self.count_tables[0]
        )
        conv1d_1_expt_0 = probound.conv1d.Conv1d.from_psam(
            psam_1, self.count_tables[0]
        )
        bmd_0_expt_0 = probound.binding.BindingMode([conv1d_0_expt_0])
        bmd_1_expt_0 = probound.binding.BindingMode([conv1d_1_expt_0])
        coop_0_0_expt_0 = probound.cooperativity.BindingCooperativity(
            probound.cooperativity.Spacing.from_psams(psam_0, psam_1),
            bmd_0_expt_0,
            bmd_1_expt_0,
        )

        conv1d_0_expt_1 = probound.conv1d.Conv1d.from_psam(
            psam_0, self.count_tables[1]
        )
        conv1d_1_expt_1 = probound.conv1d.Conv1d.from_psam(
            psam_1, self.count_tables[1]
        )
        bmd_0_expt_1 = probound.binding.BindingMode([conv1d_0_expt_1])
        bmd_1_expt_1 = probound.binding.BindingMode([conv1d_1_expt_1])
        coop_0_0_expt_1 = probound.cooperativity.BindingCooperativity(
            probound.cooperativity.Spacing.from_psams(psam_0, psam_0),
            bmd_0_expt_1,
            bmd_0_expt_1,
        )
        coop_0_1_expt_1 = probound.cooperativity.BindingCooperativity(
            probound.cooperativity.Spacing.from_psams(psam_0, psam_1),
            bmd_0_expt_1,
            bmd_1_expt_1,
        )

        round_0_expt_0 = probound.rounds.IRound()
        round_1_expt_0 = probound.rounds.BRound(
            aggregate=probound.aggregate.Aggregate(
                [
                    probound.aggregate.Contribution(
                        bmd_0_expt_0, log_alpha=0.0
                    ),
                    probound.aggregate.Contribution(
                        bmd_1_expt_0, log_alpha=0.0
                    ),
                    probound.aggregate.Contribution(
                        coop_0_0_expt_0, log_alpha=0.0
                    ),
                ]
            ),
            reference_round=round_0_expt_0,
        )
        round_2_expt_0 = probound.rounds.BURound(
            aggregate=probound.aggregate.Aggregate(
                [
                    probound.aggregate.Contribution(
                        bmd_1_expt_0, log_alpha=0.0
                    ),
                    probound.aggregate.Contribution(
                        coop_0_0_expt_0, log_alpha=0.0
                    ),
                ]
            ),
            reference_round=round_1_expt_0,
        )

        round_0_expt_1 = probound.rounds.IRound()
        round_1_expt_1 = probound.rounds.BRound(
            aggregate=probound.aggregate.Aggregate(
                [
                    probound.aggregate.Contribution(
                        coop_0_0_expt_1, log_alpha=0.0
                    ),
                    probound.aggregate.Contribution(
                        coop_0_1_expt_1, log_alpha=0.0
                    ),
                ]
            ),
            reference_round=round_0_expt_1,
        )

        self.model = probound.loss.MultiExperimentLoss(
            [
                probound.experiment.Experiment(
                    [round_0_expt_0, round_1_expt_0, round_2_expt_0]
                ),
                probound.experiment.Experiment(
                    [round_0_expt_1, round_1_expt_1]
                ),
            ]
        )

    def test_caching(self) -> None:
        with self.assertLogs("probound", "INFO") as logger:
            self.model(self.count_tables)

        calculation_counter = collections.Counter(
            [i for i in logger.output if "Calculating output" in i]
        )
        for key, val in calculation_counter.items():
            self.assertEqual(val, 1, f"'{key}' is run {val} times")

        for module in self.model.modules():
            if isinstance(module, probound.base.Component):
                for cmpts in module._blocking.values():
                    self.assertEqual(
                        len(cmpts),
                        0,
                        f"Module {module} is being blocked {module._blocking}",
                    )
                for caches in module._caches.values():
                    self.assertEqual(
                        caches,
                        (None, None),
                        f"Module {module} has cached output {module._caches}",
                    )


if __name__ == "__main__":
    unittest.main()
