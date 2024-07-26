# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring
import math
import unittest

import pandas as pd
import requests
import torch
from typing_extensions import override

import pyprobound
import pyprobound.external


class TestSingleTF(unittest.TestCase):

    @override
    def setUp(self) -> None:
        output_root = "http://pbdemo.x3dna.org/files/example_output"
        data_root = "http://pbdemo.x3dna.org/files/example_data"
        self.fit_final = requests.get(
            f"{output_root}/singleTF/fit.final.json", timeout=5
        ).json()
        self.config = requests.get(
            f"{output_root}/singleTF/config.json", timeout=5
        ).json()
        dataframes = [
            pd.read_csv(
                f"{data_root}/singleTF/countTable.0.CTCF_r3.tsv.gz",
                header=None,
                index_col=0,
                sep="\t",
            )
        ]
        self.count_tables = pyprobound.external.parse_probound_tables(
            self.fit_final, dataframes
        )
        self.model = pyprobound.external.parse_probound_model(
            self.fit_final, self.config, self.count_tables
        )

    def test_loss(self) -> None:
        with torch.inference_mode():
            nll, reg = [i.item() for i in self.model(self.count_tables)]
        java_reg = self.fit_final["metadata"]["regularization"]
        java_nll = (
            self.fit_final["metadata"]["logLikelihoodPerRead"] - java_reg
        )
        self.assertTrue(
            math.isclose(nll, java_nll, rel_tol=1e-6),
            msg=f"Calculated NLL {nll} does not match reference {java_nll}",
        )
        self.assertTrue(
            math.isclose(reg, java_reg, rel_tol=1e-6),
            msg=f"Calculated Reg {reg} does not match reference {java_reg}",
        )


class TestMultiTF(TestSingleTF):

    @override
    def setUp(self) -> None:
        output_root = "http://pbdemo.x3dna.org/files/example_output"
        data_root = "http://pbdemo.x3dna.org/files/example_data"
        self.fit_final = requests.get(
            f"{output_root}/multiTF/fit.final.json", timeout=5
        ).json()
        self.config = requests.get(
            f"{output_root}/multiTF/config.json", timeout=5
        ).json()
        dataframes = [
            pd.read_csv(
                f"{data_root}/multiTF/countTable.{fname}.tsv.gz",
                header=None,
                index_col=0,
                sep="\t",
            )
            for fname in ("0.CTCF_r3", "0.CTCF_ESAJ_TAGCGA20NGCT")
        ]
        self.count_tables = pyprobound.external.parse_probound_tables(
            self.fit_final, dataframes
        )
        self.model = pyprobound.external.parse_probound_model(
            self.fit_final, self.config, self.count_tables
        )


class TestHthExdUbx(TestSingleTF):

    @override
    def setUp(self) -> None:
        output_root = "http://pbdemo.x3dna.org/files/example_output"
        data_root = "http://pbdemo.x3dna.org/files/example_data"
        self.fit_final = requests.get(
            f"{output_root}/hthExdUbx/fit.final.json", timeout=5
        ).json()
        self.config = requests.get(
            f"{output_root}/hthExdUbx/config.json", timeout=5
        ).json()
        dataframes = [
            pd.read_csv(
                f"{data_root}/hthExdUbx/countTable.{fname}.tsv.gz",
                header=None,
                index_col=0,
                sep="\t",
            )
            for fname in (
                "0.UbxIVa-Hth-Exd.30mer1",
                "1.UbxIVa-Exd.16mer1_rep1",
                "2.UbxIVa.16mer1_rep1",
                "3.Exd",
                "4.Hth.16mer2_rep1",
            )
        ]
        self.count_tables = pyprobound.external.parse_probound_tables(
            self.fit_final, dataframes
        )
        self.model = pyprobound.external.parse_probound_model(
            self.fit_final, self.config, self.count_tables
        )


class TestEpiSELEX(TestSingleTF):

    @override
    def setUp(self) -> None:
        output_root = "http://pbdemo.x3dna.org/files/example_output"
        data_root = "http://pbdemo.x3dna.org/files/example_data"
        self.fit_final = requests.get(
            f"{output_root}/epiSELEX/fit.final.json", timeout=5
        ).json()
        self.config = requests.get(
            f"{output_root}/epiSELEX/config.json", timeout=5
        ).json()
        dataframes = [
            pd.read_csv(
                f"{data_root}/epiSELEX/countTable.{fname}.tsv.gz",
                header=None,
                index_col=0,
                sep="\t",
            )
            for fname in (
                "0.run_11_10_15__R1_ATF4_HOMODIMER_80nM_flank_PCR__None",
                "1.run_11_10_15__R1_ATF4_HOMODIMER_80nM_flank_PCR__5mCG",
                "2.run_06_05_17__R1_CEBPg_homo_75nM_lowBand__None",
                "3.run_06_05_17__R1_CEBPg_homo_75nM_lowBand__5mCG",
                "4.run_10_05_17__R1_ATF4-CEBPg_50nM_highBand__None",
                "5.run_10_05_17__R1_ATF4-CEBPg_50nM_highBand__5mCG",
            )
        ]
        self.count_tables = pyprobound.external.parse_probound_tables(
            self.fit_final, dataframes
        )
        self.model = pyprobound.external.parse_probound_model(
            self.fit_final, self.config, self.count_tables
        )


class TestMultiEpiSELEX(TestSingleTF):

    @override
    def setUp(self) -> None:
        output_root = "http://pbdemo.x3dna.org/files/example_output"
        data_root = "http://pbdemo.x3dna.org/files/example_data"
        self.fit_final = requests.get(
            f"{output_root}/multiEpiSELEX/fit.final.json", timeout=5
        ).json()
        self.config = requests.get(
            f"{output_root}/multiEpiSELEX/config.json", timeout=5
        ).json()
        dataframes = [
            pd.read_csv(
                f"{data_root}/multiEpiSELEX/countTable.{fname}.tsv.gz",
                header=None,
                index_col=0,
                sep="\t",
            )
            for fname in (
                "0.run_06_05_17__R1_CEBPg_homo_75nM_lowBand__None",
                "1.run_06_05_17__R1_CEBPg_homo_75nM_lowBand__5mCG",
                "2.run_06_05_17__R1_CEBPg_homo_75nM_lowBand__5hmC",
                "3.run_06_05_17__R1_CEBPg_homo_75nM_lowBand__6mA",
            )
        ]
        self.count_tables = pyprobound.external.parse_probound_tables(
            self.fit_final, dataframes
        )
        self.model = pyprobound.external.parse_probound_model(
            self.fit_final, self.config, self.count_tables
        )


class TestKdSingle(TestSingleTF):

    @override
    def setUp(self) -> None:
        output_root = "http://pbdemo.x3dna.org/files/example_output"
        data_root = "http://pbdemo.x3dna.org/files/example_data"
        self.fit_final = requests.get(
            f"{output_root}/KD-single/fit.final.json", timeout=5
        ).json()
        self.config = requests.get(
            f"{output_root}/KD-single/config.json", timeout=5
        ).json()
        dataframes = [
            pd.read_csv(
                f"{data_root}/KD-single/countTable.0.20201205_DlldN-12.tsv.gz",
                header=None,
                index_col=0,
                sep="\t",
            )
        ]
        self.count_tables = pyprobound.external.parse_probound_tables(
            self.fit_final, dataframes
        )
        self.model = pyprobound.external.parse_probound_model(
            self.fit_final, self.config, self.count_tables
        )


class TestKdMulti(TestSingleTF):

    @override
    def setUp(self) -> None:
        output_root = "http://pbdemo.x3dna.org/files/example_output"
        data_root = "http://pbdemo.x3dna.org/files/example_data"
        self.fit_final = requests.get(
            f"{output_root}/KD-multi/fit.final.json", timeout=5
        ).json()
        self.config = requests.get(
            f"{output_root}/KD-multi/config.json", timeout=5
        ).json()
        dataframes = [
            pd.read_csv(
                f"{data_root}/KD-multi/countTable.{fname}.tsv.gz",
                header=None,
                index_col=0,
                sep="\t",
            )
            for fname in (
                "0.20201205_DlldN-9",
                "1.20201205_DlldN-10",
                "2.20201205_DlldN-11",
                "3.20201205_DlldN-12",
            )
        ]
        self.count_tables = pyprobound.external.parse_probound_tables(
            self.fit_final, dataframes
        )
        self.model = pyprobound.external.parse_probound_model(
            self.fit_final, self.config, self.count_tables
        )


class TestRBP(TestSingleTF):

    @override
    def setUp(self) -> None:
        output_root = "http://pbdemo.x3dna.org/files/example_output"
        data_root = "http://pbdemo.x3dna.org/files/example_data"
        self.fit_final = requests.get(
            f"{output_root}/RBP/fit.final.json", timeout=5
        ).json()
        self.config = requests.get(
            f"{output_root}/RBP/config.json", timeout=5
        ).json()
        dataframes = [
            pd.read_csv(
                f"{data_root}/RBP/countTable.{fname}.tsv.gz",
                header=None,
                index_col=0,
                sep="\t",
            )
            for fname in (
                "0.RBFOX2-1nM",
                "1.RBFOX2-4nM",
                "2.RBFOX2-14nM",
                "3.RBFOX2-40nM",
                "4.RBFOX2-121nM",
                "5.RBFOX2-365nM",
                "6.RBFOX2-1100nM",
                "7.RBFOX2-3300nM",
                "8.RBFOX2-9800nM",
            )
        ]
        self.count_tables = pyprobound.external.parse_probound_tables(
            self.fit_final, dataframes
        )
        self.model = pyprobound.external.parse_probound_model(
            self.fit_final, self.config, self.count_tables
        )


class TestChIPSingle(TestSingleTF):

    @override
    def setUp(self) -> None:
        output_root = "http://pbdemo.x3dna.org/files/example_output"
        data_root = "http://pbdemo.x3dna.org/files/example_data"
        self.fit_final = requests.get(
            f"{output_root}/ChIP-single/fit.final.json", timeout=5
        ).json()
        self.config = requests.get(
            f"{output_root}/ChIP-single/config.json", timeout=5
        ).json()
        dataframes = [
            pd.read_csv(
                f"{data_root}/ChIP-single/"
                "countTable.0.IMR90_GR_chip-seq_rep1.tsv.gz",
                header=None,
                index_col=0,
                sep="\t",
            )
        ]
        self.count_tables = pyprobound.external.parse_probound_tables(
            self.fit_final, dataframes
        )
        self.model = pyprobound.external.parse_probound_model(
            self.fit_final, self.config, self.count_tables
        )


class TestChIPMulti(TestSingleTF):

    @override
    def setUp(self) -> None:
        output_root = "http://pbdemo.x3dna.org/files/example_output"
        data_root = "http://pbdemo.x3dna.org/files/example_data"
        self.fit_final = requests.get(
            f"{output_root}/ChIP-multi/fit.final.json", timeout=5
        ).json()
        self.config = requests.get(
            f"{output_root}/ChIP-multi/config.json", timeout=5
        ).json()
        dataframes = [
            pd.read_csv(
                f"{data_root}/ChIP-multi/countTable.{fname}.tsv.gz",
                header=None,
                index_col=0,
                sep="\t",
            )
            for fname in ("0.GR_30", "1.GR_300", "2.GR_3000")
        ]
        self.count_tables = pyprobound.external.parse_probound_tables(
            self.fit_final, dataframes
        )
        self.model = pyprobound.external.parse_probound_model(
            self.fit_final, self.config, self.count_tables
        )


class TestKinase(TestSingleTF):

    @override
    def setUp(self) -> None:
        output_root = "http://pbdemo.x3dna.org/files/example_output"
        data_root = "http://pbdemo.x3dna.org/files/example_data"
        self.fit_final = requests.get(
            f"{output_root}/Kinase/fit.final.json", timeout=5
        ).json()
        self.config = requests.get(
            f"{output_root}/Kinase/config.json", timeout=5
        ).json()
        dataframes = [
            pd.read_csv(
                f"{data_root}/Kinase/countTable.{fname}.tsv.gz",
                header=None,
                index_col=0,
                sep="\t",
            )
            for fname in (
                "0.200205_Src-Kinase_5m",
                "1.200205_Src-Kinase_20m",
                "2.200205_Src-Kinase_60m",
            )
        ]
        self.count_tables = pyprobound.external.parse_probound_tables(
            self.fit_final, dataframes
        )
        self.model = pyprobound.external.parse_probound_model(
            self.fit_final, self.config, self.count_tables
        )


if __name__ == "__main__":
    unittest.main()
