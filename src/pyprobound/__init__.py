"""Python implementation of ProBound."""

# pylint: disable=undefined-variable

import torch

__version__ = "1.4.0"
__precision__ = torch.float32
__all__ = [
    "alphabets",
    "layers",
    "rounds",
    "Aggregate",
    "Contribution",
    "Binding",
    "BindingOptim",
    "Call",
    "Component",
    "Spec",
    "Step",
    "Transform",
    "Mode",
    "Cooperativity",
    "Spacing",
    "Experiment",
    "BaseLoss",
    "Loss",
    "MultiExperimentLoss",
    "MultiRoundMSLELoss",
    "Optimizer",
    "Batch",
    "CountBatch",
    "CountTable",
    "EvenSampler",
    "MultitaskLoader",
    "Table",
    "get_dataframe",
    "sample_counts",
    "sample_dataframe",
    "score",
]
# not re-exported from pyprobound: alphabets, layers, rounds
# must be imported separately: containers, external, fitting, plotting, utils

from . import alphabets, layers, rounds
from .aggregate import Aggregate, Contribution
from .base import Binding, BindingOptim, Call, Component, Spec, Step, Transform
from .cooperativity import Cooperativity, Spacing
from .experiment import Experiment
from .loss import BaseLoss, Loss, MultiExperimentLoss, MultiRoundMSLELoss
from .mode import Mode
from .optimizer import Optimizer
from .table import (
    Batch,
    CountBatch,
    CountTable,
    EvenSampler,
    MultitaskLoader,
    Table,
    get_dataframe,
    sample_counts,
    sample_dataframe,
    score,
)

del aggregate, base, mode, cooperativity, experiment, loss, optimizer, table  # type: ignore[name-defined]
