# pylint: disable=missing-function-docstring
"""Test suite for ProBound"""
import random

import numpy as np
import pandas as pd
import torch

import pyprobound

torch.set_grad_enabled(False)


def make_count_table(
    alphabet: pyprobound.alphabets.Alphabet = pyprobound.alphabets.DNA(),
    n_columns: int = 2,
    n_seqs: int = 100,
    min_input_length: int = 20,
    max_input_length: int = 24,
    left_flank: str = "ACGTACGT",
    right_flank: str = "ACGTACGT",
) -> pyprobound.CountTable:
    seqs = [
        "".join(
            random.choices(
                alphabet.alphabet + ("*",),
                k=random.randint(min_input_length, max_input_length),
            )
        )
        for _ in range(n_seqs)
    ]

    df = pd.DataFrame(index=seqs, data=np.ones(shape=(len(seqs), n_columns)))

    return pyprobound.CountTable(
        dataframe=df,
        alphabet=alphabet,
        left_flank=left_flank,
        right_flank=right_flank,
    )
