"""Re-implementation of ProBound in PyTorch"""

import torch

__version__ = "0.1.0"
__precision__ = torch.float32

from .aggregate import Aggregate, Contribution
from .alphabet import DNA, RNA, Alphabet, Codon, Protein
from .binding import BindingMode
from .conv1d import Conv0d, Conv1d
from .cooperativity import BindingCooperativity, Spacing
from .experiment import Experiment
from .layers import MaxPool1d, MaxPool1dSpec
from .loss import MultiExperimentLoss
from .optimizer import Optimizer
from .psam import PSAM, NonSpecific
from .rounds import BRound, BRRound, BURound, ERound, FRound, IRound
from .table import (
    CountTable,
    get_dataframe,
    sample_counts,
    sample_dataframe,
    score,
)

# fmt: off
# pylint: disable-next=undefined-variable
# del aggregate, alphabet, binding, cooperativity, experiment, loss, optimizer, psam, rounds, table # type: ignore[name-defined]
# fmt: on
