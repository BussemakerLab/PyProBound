# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
"""Tests for paired-end ATAC count tables and fragment rounds."""
import unittest

import numpy as np
import pandas as pd
import torch
from typing_extensions import override

import pyprobound
from pyprobound.rounds import AtacFragmentBoundRound, AtacFragmentRound


def make_paired_count_table(
    n_seqs: int = 32,
    seq_length: int = 20,
    n_columns: int = 2,
) -> pyprobound.PairedCountTable:
    alphabet = pyprobound.alphabets.DNA()
    letters = list(alphabet.alphabet)
    left = [
        "".join(np.random.choice(letters, size=seq_length))
        for _ in range(n_seqs)
    ]
    right = [
        "".join(np.random.choice(letters, size=seq_length))
        for _ in range(n_seqs)
    ]
    index = pd.MultiIndex.from_arrays(
        [left, right], names=["sequence_left", "sequence_right"]
    )
    df = pd.DataFrame(
        index=index,
        data=np.random.poisson(lam=2.0, size=(n_seqs, n_columns)).astype(float)
        + 1.0,
        columns=[f"R{i}" for i in range(n_columns)],
    )
    return pyprobound.PairedCountTable(dataframe=df, alphabet=alphabet)


class TestPairedAtac(unittest.TestCase):
    @override
    def setUp(self) -> None:
        torch.set_grad_enabled(True)
        self.count_table = make_paired_count_table()
        self.alphabet = self.count_table.alphabet

    def test_paired_table_shapes(self) -> None:
        self.assertEqual(len(self.count_table), 32)
        self.assertEqual(self.count_table.seqs_left.shape[0], 32)
        self.assertEqual(self.count_table.seqs_right.shape[0], 32)
        self.assertEqual(
            self.count_table.seqs_left.shape[-1],
            self.count_table.seqs_right.shape[-1],
        )
        batch = self.count_table[0]
        self.assertIsInstance(batch, pyprobound.PairedCountBatch)
        self.assertEqual(len(batch.seqs), 2)

    def test_product_enrichment_matches_sum_of_logs(self) -> None:
        nonspecific = pyprobound.layers.NonSpecific(
            alphabet=self.alphabet, name="NS"
        )
        mode = pyprobound.Mode.from_nonspecific(nonspecific, self.count_table)
        initial = pyprobound.rounds.InitialRound()
        r1 = AtacFragmentRound.from_binding([mode], initial)
        for ctrb in r1.aggregate.contributions:
            r1.aggregate.activity_heuristic(ctrb)
        seqs = self.count_table.seqs
        n_batch = seqs[0].shape[0]
        log_e = r1.log_enrichment(seqs)
        expected = r1.aggregate(torch.cat([seqs[0], seqs[1]], dim=0))
        expected = expected[:n_batch] + expected[n_batch:]
        self.assertTrue(torch.allclose(log_e, expected))
        self.assertTrue(torch.isfinite(log_e).all())

    def test_experiment_loss_smoke(self) -> None:
        nonspecific = pyprobound.layers.NonSpecific(
            alphabet=self.alphabet, name="NS"
        )
        mode = pyprobound.Mode.from_nonspecific(nonspecific, self.count_table)
        initial = pyprobound.rounds.InitialRound()
        r1 = AtacFragmentRound.from_binding([mode], initial)
        for ctrb in r1.aggregate.contributions:
            r1.aggregate.activity_heuristic(ctrb)
        experiment = pyprobound.Experiment(
            [initial, r1],
            counts_per_round=self.count_table.counts_per_round,
        )
        model = pyprobound.MultiExperimentLoss([experiment])
        loss = model([self.count_table])
        self.assertTrue(torch.isfinite(loss.negloglik))
        loss.negloglik.backward()

    def test_saturated_product_round(self) -> None:
        nonspecific = pyprobound.layers.NonSpecific(
            alphabet=self.alphabet, name="NS"
        )
        mode = pyprobound.Mode.from_nonspecific(nonspecific, self.count_table)
        initial = pyprobound.rounds.InitialRound()
        r1 = AtacFragmentBoundRound.from_binding([mode], initial)
        for ctrb in r1.aggregate.contributions:
            r1.aggregate.activity_heuristic(ctrb)
        seqs = self.count_table.seqs
        n_batch = seqs[0].shape[0]
        log_e = r1.log_enrichment(seqs)
        log_z = r1.aggregate(torch.cat([seqs[0], seqs[1]], dim=0))
        expected = torch.nn.functional.logsigmoid(
            log_z[:n_batch]
        ) + torch.nn.functional.logsigmoid(log_z[n_batch:])
        self.assertTrue(torch.allclose(log_e, expected))

    def test_get_paired_dataframe_roundtrip(self) -> None:
        path = "/tmp/pyprobound_paired_smoke.tsv"
        left = ["ACGT" * 5, "TTTT" * 5]
        right = ["GGGG" * 5, "AAAA" * 5]
        pd.DataFrame(
            {
                "sequence_left": left,
                "sequence_right": right,
                "R0": [1, 2],
                "R1": [3, 4],
            }
        ).to_csv(path, sep="\t", index=False)
        df = pyprobound.get_paired_dataframe(path)
        table = pyprobound.PairedCountTable(
            dataframe=df, alphabet=pyprobound.alphabets.DNA()
        )
        self.assertEqual(len(table), 2)
        self.assertEqual(list(table.counts_per_round.tolist()), [3.0, 7.0])


if __name__ == "__main__":
    unittest.main()
