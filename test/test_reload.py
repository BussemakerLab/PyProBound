# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
import functools
import os
import tempfile
import unittest

import torch
from typing_extensions import override

import pyprobound

from . import make_count_table


def make_experiment(
    kernel_size: int = 4, pairwise_distance: int = 0
) -> tuple[
    pyprobound.CountTable,
    pyprobound.layers.PSAM,
    pyprobound.rounds.BoundRound,
    pyprobound.Experiment,
]:
    count_table = make_count_table(n_columns=3)
    psam = pyprobound.layers.PSAM(
        kernel_size=kernel_size,
        alphabet=count_table.alphabet,
        pairwise_distance=pairwise_distance,
    )
    mode = pyprobound.Mode.from_psam(psam, count_table)
    initial = pyprobound.rounds.InitialRound()
    bound = pyprobound.rounds.BoundRound.from_binding([mode], initial)
    unbound = pyprobound.rounds.UnboundRound.from_round(bound)
    experiment = pyprobound.Experiment([initial, bound, unbound])
    return count_table, psam, bound, experiment


def randomize(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.data = torch.randn_like(param)


class TestReload(unittest.TestCase):
    @override
    def setUp(self) -> None:
        fd, self.checkpoint = tempfile.mkstemp(suffix=".pt")
        os.close(fd)

    @override
    def tearDown(self) -> None:
        os.remove(self.checkpoint)

    def test_reload_same_shape(self) -> None:
        count_table, _, _, saved = make_experiment()
        randomize(saved)
        expected = saved(count_table.seqs)
        saved.save(self.checkpoint)

        _, _, _, loaded = make_experiment()
        loaded.reload(self.checkpoint)
        torch.testing.assert_close(loaded(count_table.seqs), expected)

    def test_reload_changed_footprint(self) -> None:
        count_table, psam, _, saved = make_experiment(kernel_size=4)
        psam.update_footprint(left_shift=1, right_shift=2)
        randomize(saved)
        expected = saved(count_table.seqs)
        saved.save(self.checkpoint)

        _, loaded_psam, _, loaded = make_experiment(kernel_size=4)
        loaded.reload(self.checkpoint)
        self.assertEqual(loaded_psam.kernel_size, psam.kernel_size)
        torch.testing.assert_close(loaded(count_table.seqs), expected)

    def test_reload_shrunk_footprint(self) -> None:
        count_table, psam, _, saved = make_experiment(kernel_size=6)
        psam.update_footprint(left_shift=-1, right_shift=-2)
        randomize(saved)
        expected = saved(count_table.seqs)
        saved.save(self.checkpoint)

        _, loaded_psam, _, loaded = make_experiment(kernel_size=6)
        loaded.reload(self.checkpoint)
        self.assertEqual(loaded_psam.kernel_size, 3)
        torch.testing.assert_close(loaded(count_table.seqs), expected)

    def test_reload_changed_footprint_pairwise(self) -> None:
        count_table, psam, _, saved = make_experiment(
            kernel_size=4, pairwise_distance=2
        )
        psam.update_footprint(left_shift=1, right_shift=1)
        randomize(saved)
        expected = saved(count_table.seqs)
        saved.save(self.checkpoint)

        _, loaded_psam, _, loaded = make_experiment(
            kernel_size=4, pairwise_distance=2
        )
        loaded.reload(self.checkpoint)
        self.assertEqual(loaded_psam.kernel_size, psam.kernel_size)
        self.assertEqual(set(loaded_psam.betas.keys()), set(psam.betas.keys()))
        torch.testing.assert_close(loaded(count_table.seqs), expected)

    def test_shared_parameter_stays_shared(self) -> None:
        count_table, _, bound, saved = make_experiment()
        unbound = saved.rounds[-1]
        unbound.log_depth = bound.log_depth
        randomize(saved)
        expected = saved(count_table.seqs)
        saved.save(self.checkpoint)

        _, _, loaded_bound, loaded = make_experiment()
        loaded_unbound = loaded.rounds[-1]
        loaded_unbound.log_depth = loaded_bound.log_depth
        loaded.reload(self.checkpoint)
        self.assertIs(loaded_unbound.log_depth, loaded_bound.log_depth)
        torch.testing.assert_close(loaded(count_table.seqs), expected)

    def test_shared_parameter_stays_shared_after_footprint_change(
        self,
    ) -> None:
        count_table, psam, bound, saved = make_experiment(kernel_size=4)
        saved.rounds[-1].log_depth = bound.log_depth
        psam.update_footprint(left_shift=2, right_shift=1)
        randomize(saved)
        expected = saved(count_table.seqs)
        saved.save(self.checkpoint)

        _, _, loaded_bound, loaded = make_experiment(kernel_size=4)
        loaded.rounds[-1].log_depth = loaded_bound.log_depth
        loaded.reload(self.checkpoint)
        self.assertIs(loaded.rounds[-1].log_depth, loaded_bound.log_depth)
        torch.testing.assert_close(loaded(count_table.seqs), expected)

    def test_requires_grad_preserved(self) -> None:
        _, _, _, saved = make_experiment()
        saved.save(self.checkpoint)

        _, _, loaded_bound, loaded = make_experiment()
        loaded.freeze()
        loaded_bound.log_depth.requires_grad_(True)
        loaded.reload(self.checkpoint)
        self.assertTrue(loaded_bound.log_depth.requires_grad)
        self.assertFalse(loaded.rounds[-1].log_depth.requires_grad)

    def test_shared_parameter_conflicting_values(self) -> None:
        _, _, bound, model = make_experiment()
        model.rounds[-1].log_depth = bound.log_depth
        state_dict = model.state_dict()
        shared_keys = [
            key
            for key in state_dict
            if functools.reduce(getattr, key.split("."), model)
            is bound.log_depth
        ]
        self.assertGreater(len(shared_keys), 1)
        for value, key in enumerate(shared_keys):
            state_dict[key] = torch.tensor(float(value))
        with self.assertRaises(ValueError):
            model.reload_from_state_dict(state_dict)


if __name__ == "__main__":
    unittest.main()
