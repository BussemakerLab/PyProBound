# pylint: disable=missing-class-docstring, missing-module-docstring
import json
import math
import unittest

import pandas as pd
import requests
import torch
from typing_extensions import override

import probound.alphabet
import probound.binding
import probound.conv1d
import probound.experiment
import probound.loss
import probound.psam
import probound.table


class TestDll(unittest.TestCase):
    @override
    def setUp(self) -> None:
        # read in table
        dataframe = pd.read_csv(
            "http://pbdemo.x3dna.org/files/example_data/"
            "KD-single/countTable.0.20201205_DlldN-12.tsv.gz",
            header=None,
            index_col=0,
            sep="\t",
        )
        alphabet = probound.alphabet.DNA()
        self.count_tables = [
            probound.table.CountTable(
                dataframe,
                alphabet,
                left_flank="GAGTTCTACAGTCCGACCTGG",
                right_flank="CCAGGACTCGGACCTGGA",
                left_flank_length=6,
                right_flank_length=6,
            )
        ]

        # specify model
        nonspecific = probound.psam.NonSpecific(alphabet=alphabet)
        psam = probound.psam.PSAM(
            kernel_size=10,
            alphabet=alphabet,
            interaction_distance=9,
            normalize=False,
        )
        conv0d = probound.conv1d.Conv0d.from_nonspecific(
            nonspecific, self.count_tables[0]
        )
        conv1d = probound.conv1d.Conv1d.from_psam(psam, self.count_tables[0])
        binding_modes = [
            probound.binding.BindingMode([conv0d]),
            probound.binding.BindingMode([conv1d]),
        ]
        i_round = probound.rounds.IRound()
        b_round = probound.rounds.BRound.from_binding(
            binding_modes,
            i_round,
            target_concentration=100,
            library_concentration=20,
        )
        f_round = probound.rounds.FRound.from_round(b_round)
        experiment = probound.experiment.Experiment(
            [i_round, b_round, f_round],
            counts_per_round=self.count_tables[0].counts_per_round,
        )
        self.model = probound.loss.MultiExperimentLoss(
            [experiment],
            lambda_l2=1e-6,
            pseudocount=200,
            exponential_bound=40,
            full_loss=False,
        )

        # read in parameters
        self.json = json.loads(
            requests.get(
                "http://pbdemo.x3dna.org/files/example_output/"
                "KD-single/fit.final.json",
                timeout=5,
            ).text
        )

        # fill in log_eta
        for rnd, log_eta in zip(
            experiment.rounds,
            self.json["coefficients"]["countTable"][0]["h"],
            strict=True,
        ):
            torch.nn.init.constant_(rnd.log_eta, log_eta)

        # fill in log_alpha
        torch.nn.init.constant_(
            b_round.aggregate.contributions[0].log_alpha,
            self.json["coefficients"]["bindingModes"][0]["activity"][0][0]
            - math.log(self.count_tables[0].input_shape),
        )
        torch.nn.init.constant_(
            b_round.aggregate.contributions[1].log_alpha,
            self.json["coefficients"]["bindingModes"][1]["activity"][0][0],
        )

        # fill in betas
        mono = self.json["coefficients"]["bindingModes"][1]["mononucleotide"]
        di = self.json["coefficients"]["bindingModes"][1]["dinucleotide"]
        alphalen = len(alphabet.alphabet)
        for key, param in psam.betas.items():
            elements = [int(i) for i in key.split("-")]
            elements = [i - 1 for i in elements[:-1]] + elements[-1:]
            if len(elements) == 2:
                torch.nn.init.constant_(
                    param, mono[elements[0] * alphalen + elements[1]]
                )
            else:
                torch.nn.init.constant_(
                    param,
                    di[elements[1] - elements[0] - 1][
                        elements[0] * (alphalen**2) + elements[2]
                    ],
                )

    def test_loss(self) -> None:
        with torch.inference_mode():
            nll, reg = [i.item() for i in self.model(self.count_tables)]
        java_reg = self.json["metadata"]["regularization"]
        java_nll = self.json["metadata"]["logLikelihoodPerRead"] - java_reg
        self.assertAlmostEqual(
            nll,
            java_nll,
            places=6,
            msg=f"Calculated NLL {nll} does not match reference {java_nll}",
        )
        self.assertAlmostEqual(
            reg,
            java_reg,
            places=3,
            msg=f"Calculated Reg {reg} does not match reference {java_reg}",
        )


if __name__ == "__main__":
    unittest.main()
