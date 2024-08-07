# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring, protected-access
import collections
import os
import tempfile
import unittest
from typing import Any

from torch import Tensor
from typing_extensions import override

import pyprobound

from . import make_count_table


class TestCaching(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.count_tables = [
            make_count_table(n_columns=3),
            make_count_table(n_columns=2),
        ]

        psam_0 = pyprobound.layers.PSAM(1, self.count_tables[0].alphabet)
        psam_1 = pyprobound.layers.PSAM(1, self.count_tables[0].alphabet)
        bmd_0_expt_0 = pyprobound.Mode.from_psam(psam_0, self.count_tables[0])
        bmd_1_expt_0 = pyprobound.Mode.from_psam(psam_1, self.count_tables[0])
        coop_0_0_expt_0 = pyprobound.Cooperativity(
            pyprobound.Spacing([psam_0], [psam_1]), bmd_0_expt_0, bmd_1_expt_0
        )

        bmd_0_expt_1 = pyprobound.Mode.from_psam(psam_0, self.count_tables[1])
        bmd_1_expt_1 = pyprobound.Mode.from_psam(psam_1, self.count_tables[1])
        coop_0_0_expt_1 = pyprobound.Cooperativity(
            pyprobound.Spacing([psam_0], [psam_0]), bmd_0_expt_1, bmd_0_expt_1
        )
        coop_0_1_expt_1 = pyprobound.Cooperativity(
            pyprobound.Spacing([psam_0], [psam_1]), bmd_0_expt_1, bmd_1_expt_1
        )

        round_0_expt_0 = pyprobound.rounds.InitialRound()
        round_1_expt_0 = pyprobound.rounds.BoundRound(
            aggregate=pyprobound.Aggregate(
                [
                    pyprobound.Contribution(bmd_0_expt_0, log_activity=0.0),
                    pyprobound.Contribution(bmd_1_expt_0, log_activity=0.0),
                    pyprobound.Contribution(coop_0_0_expt_0, log_activity=0.0),
                ]
            ),
            reference_round=round_0_expt_0,
        )
        round_2_expt_0 = pyprobound.rounds.BoundUnsaturatedRound(
            aggregate=pyprobound.Aggregate(
                [
                    pyprobound.Contribution(bmd_1_expt_0, log_activity=0.0),
                    pyprobound.Contribution(coop_0_0_expt_0, log_activity=0.0),
                ]
            ),
            reference_round=round_1_expt_0,
        )

        round_0_expt_1 = pyprobound.rounds.InitialRound()
        round_1_expt_1 = pyprobound.rounds.BoundRound(
            aggregate=pyprobound.Aggregate(
                [
                    pyprobound.Contribution(coop_0_0_expt_1, log_activity=0.0),
                    pyprobound.Contribution(coop_0_1_expt_1, log_activity=0.0),
                ]
            ),
            reference_round=round_0_expt_1,
        )

        self.model = pyprobound.MultiExperimentLoss(
            [
                pyprobound.Experiment(
                    [round_0_expt_0, round_1_expt_0, round_2_expt_0]
                ),
                pyprobound.Experiment([round_0_expt_1, round_1_expt_1]),
            ]
        )

        # Give a unique name to every submodule
        for idx, module in enumerate(self.model.modules()):
            if isinstance(module, pyprobound.Component):
                module.name = str(idx)

    def test_caching(self) -> None:
        with self.assertLogs("pyprobound", "INFO") as logger:
            self.model(self.count_tables)

        calculation_counter = collections.Counter(
            [i for i in logger.output if "Calculating output" in i]
        )
        for key, val in calculation_counter.items():
            self.assertEqual(val, 1, f"'{key}' is run {val} times")

        for module in self.model.modules():
            if isinstance(module, pyprobound.Component):
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

    def check_forward(self, optimizer: pyprobound.Optimizer[Any]) -> None:
        try:
            optimizer.model(optimizer.train_tables)
        except RuntimeError:
            self.fail("RuntimeError raised during forward calculation")

    def test_reload(self) -> None:
        greedy_fd, greedy_checkpoint = None, None
        try:
            greedy_fd, greedy_checkpoint = tempfile.mkstemp()
            optimizer = pyprobound.Optimizer(
                self.model,
                self.count_tables,
                device="cpu",
                checkpoint=greedy_checkpoint,
            )

            # Initial forward
            optimizer.save(optimizer.checkpoint)
            self.check_forward(optimizer)

            # Update footprints
            old_symmetries: list[Tensor] = []
            old_flanks: list[tuple[int, int]] = []
            for module in optimizer.model.modules():
                if isinstance(module, pyprobound.layers.PSAM):
                    old_symmetries.append(module.symmetry)
                    module.update_footprint(1, 1)
            for table in optimizer.train_tables:
                old_flanks.append(
                    (table.left_flank_length, table.right_flank_length)
                )
            optimizer.update_read_length(
                [
                    pyprobound.Call(
                        module,
                        "update_read_length",
                        {"left_shift": 1, "right_shift": 1},
                    )
                    for module in optimizer.model.modules()
                    if isinstance(module, pyprobound.Mode)
                ]
            )
            self.check_forward(optimizer)

            # Reload
            optimizer.reload(optimizer.checkpoint)
            new_symmetries: list[Tensor] = []
            new_flanks: list[tuple[int, int]] = []
            for module in optimizer.model.modules():
                if isinstance(module, pyprobound.layers.PSAM):
                    new_symmetries.append(module.symmetry)
            for table in optimizer.train_tables:
                new_flanks.append(
                    (table.left_flank_length, table.right_flank_length)
                )
            self.check_forward(optimizer)
            self.assertEqual(
                old_symmetries,
                new_symmetries,
                "Symmetries have changed after reloading",
            )
            self.assertEqual(
                old_flanks, new_flanks, "Flanks have changed after reloading"
            )
        finally:
            if greedy_fd is not None:
                os.close(greedy_fd)
            if greedy_checkpoint is not None:
                os.remove(greedy_checkpoint)


if __name__ == "__main__":
    unittest.main()
