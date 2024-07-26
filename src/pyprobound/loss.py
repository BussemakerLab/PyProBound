"""Loss components.

Members are explicitly re-exported in pyprobound.
"""

import abc
from collections.abc import Iterable, Iterator
from typing import Generic, Literal, NamedTuple, TypeVar, cast

import torch
from torch import Tensor
from typing_extensions import override

from . import __precision__
from .base import Component, Spec, Transform
from .containers import TModuleList
from .cooperativity import Cooperativity
from .experiment import Experiment
from .layers import PSAM
from .mode import Mode
from .rounds import BaseRound
from .table import CountBatch
from .utils import get_ordinal, get_split_size

T = TypeVar("T")


class Loss(NamedTuple):
    """A loss value, interpreted as the sum of both elements.

    Attributes:
        negloglik: The negative log likelihood.
        regularization: The regularization value.
    """

    negloglik: Tensor
    regularization: Tensor


class LossModule(Component, Generic[T]):
    """Component that calculates loss.

    See https://github.com/pytorch/pytorch/issues/45414
    """

    @override
    @abc.abstractmethod
    def components(self) -> Iterator[Transform]: ...

    @override
    @abc.abstractmethod
    def forward(self, batch: Iterable[T]) -> Loss:
        """Calculates the loss."""

    @override
    def __call__(self, batch: Iterable[T]) -> Loss:
        return cast(Loss, super().__call__(batch))

    @abc.abstractmethod
    def get_setup_string(self) -> str:
        """A description used when printing the output of an optimizer."""


class MultiExperimentLoss(LossModule[CountBatch]):
    """Multitask optimization of multiple count tables with a Poisson loss.

    Attributes:
        experiments (TModuleList[Experiment]): The experiments to be jointly
            optimized.
    """

    def __init__(
        self,
        experiments: Iterable[Experiment],
        weights: Iterable[float] | None = None,
        lambda_l2: float = 1e-6,
        lambda_l1: float = 0,
        pseudocount: float = 0,
        exponential_bound: float = 40,
        full_loss: bool = False,
        dilute_regularization: bool = False,
        exclude_regularization: Iterable[str] = tuple(),
        equalize_contribution: bool = False,
        max_split: int | None = None,
    ) -> None:
        """Initializes the multitask loss from an iterable of experiments.

        Args:
            experiments: The experiments to be jointly optimized.
            weights: Multiplier to the NLL of each corresponding experiment.
            lambda_l2: L2 regularization hyperparameter.
            lambda_l1: L1 regularization hyperparameter.
            pseudocount: Scaling factor for Dirichlet-inspired regularization.
            full_loss: Whether to compute the constant terms of the NLL.
            dilute_regularization: Whether to keep hyperparameters fixed as the
                number of experiments increases.
            exclude_regularization: Keywords used to exclude parameters from
                regularization.
            equalize_contribution: Whether to update the weights so that the
                rescaled losses are constant relative to each other.
            max_split: Maximum number of sequences scored at a time.
        """
        super().__init__(name="")

        # Store loss attributes
        self.experiments: TModuleList[Experiment] = TModuleList(experiments)
        self.dilute_regularization = dilute_regularization
        self.exclude_regularization = tuple(exclude_regularization)
        self.full_loss = full_loss
        self.equalize_contribution = equalize_contribution
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.pseudocount = pseudocount
        self.exponential_bound = exponential_bound
        self.max_split = max_split

        # Store scaling factor for loss of each experiment
        if weights is None:
            weights = [1 / len(self.experiments)] * len(self.experiments)
        else:
            weights = list(weights)
        self.weights = weights
        if len(self.weights) != len(self.experiments):
            raise ValueError(
                f"Length of weights {len(self.weights)} does not match"
                f" number of experiments {len(self.experiments)}"
            )

        # Fill in spec names for each type
        type_to_specs: dict[str, dict[Spec, None]] = {}
        for mod in self.modules():
            if isinstance(mod, Spec):
                type_name = type(mod).__name__
                if type_name not in type_to_specs:
                    type_to_specs[type_name] = {}
                type_to_specs[type_name][mod] = None
        for type_specs in type_to_specs.values():
            for spec_idx, spec in enumerate(type_specs):
                if spec.name == "":
                    spec.name = get_ordinal(spec_idx)

        # Add ancestry information to component names
        for expt_idx, expt in enumerate(self.components()):
            if expt.name == "":
                expt.name = get_ordinal(expt_idx)
            for rnd_idx, rnd in enumerate(expt.components()):
                if rnd.name == "":
                    rnd.name = f"{expt}→{get_ordinal(rnd_idx)}"
                for agg in rnd.components():
                    agg.name = f"{rnd}→{agg.name}"
                    for ctrb_idx, ctrb in enumerate(agg.components()):
                        ctrb.name = f"{agg}→{get_ordinal(ctrb_idx)}"
            for mod in expt.modules():
                if isinstance(mod, Mode):
                    if mod.name == "":
                        mod.name = f"{expt}→" + "-".join(
                            str(i) for i in mod.key()
                        )
                    for layer_idx, layer in enumerate(mod.layers):
                        if layer.name == "":
                            layer.name = (
                                f"{mod}→Layer{layer_idx}:{layer.layer_spec}←"
                            )
                elif isinstance(mod, Cooperativity):
                    if mod.spacing.name == "":
                        mod.spacing.name = (
                            "-".join(str(i) for i in mod.spacing.mode_key_a)
                            + "::"
                            + "-".join(str(i) for i in mod.spacing.mode_key_b)
                        )
                    if mod.name == "":
                        mod.name = f"{expt}→{mod.spacing.name}"

        # Check that all names are unique
        specs = [mod for mod in self.modules() if isinstance(mod, Spec)]
        if len(specs) != len(set(str(spec) for spec in specs)):
            raise ValueError("Binding component names are not unique")
        if len(self.experiments) != len(
            set(expt.name for expt in self.experiments)
        ):
            raise ValueError("Experiment names are not unique")
        rnds = [mod for mod in self.modules() if isinstance(mod, BaseRound)]
        if len(rnds) != len(set(rnd.name for rnd in rnds)):
            raise ValueError("Round names are not unique")

    @override
    def components(self) -> Iterator[Experiment]:
        return iter(self.experiments)

    @override
    def get_setup_string(self) -> str:
        out: list[str] = []
        out.extend(
            [
                "### Regularization:",
                f"\t L1 Lambda: {self.lambda_l1}",
                f"\t L2 Lambda: {self.lambda_l2}",
                f"\t Pseudocount: {self.pseudocount}",
                f"\t Exponential Bound: {self.exponential_bound}",
                f"\t Excluded Reg.: {self.exclude_regularization}",
                f"\t Eq. Contribution: {self.equalize_contribution}",
                f"\t Weights: {self.weights}",
            ]
        )

        out.append("\n### Experiments:")
        for expt in self.experiments:
            round_format = [
                str(rnd) if rnd in expt.observed_rounds else f"({str(rnd)})"
                for rnd in expt.rounds
            ]
            out.extend(
                [
                    f"\tExperiment: {str(expt)}",
                    f"\t\tRounds: [{', '.join(round_format)}]",
                ]
            )

        out.append("\n### Binding Components:")
        for binding_idx, (binding, optim) in enumerate(
            self.optim_procedure().items()
        ):
            out.extend([f"\t Mode {binding_idx}: {binding}"])
            for ancestors in optim.ancestry:
                out.append(f"\t\t{ancestors[-2]}")

        return "\n".join(out)

    def regularization(self, experiment: Experiment | None = None) -> Tensor:
        """Calculates parameter regularization.

        Args:
            experiment: The experiment containing parameters to be regularized.
                None defaults to dilution over all parameters.

        Returns:
            The regularization value as a scalar tensor.
        """

        # Get flattened parameter vector
        param_list = []
        model = self if experiment is None else experiment
        for name, param in model.named_parameters():
            if torch.any(torch.isneginf(param)):
                continue
            if any(exclude in name for exclude in self.exclude_regularization):
                continue
            param_list.append(param.flatten())
        param_vec = torch.cat(param_list)
        regularization = torch.tensor(0.0, device=param_vec.device)

        # L2 regularization
        if self.lambda_l2 > 0:
            regularization += self.lambda_l2 * param_vec.square().sum()

        # L1 regularization
        if self.lambda_l1 > 0:
            regularization += self.lambda_l1 * param_vec.abs().sum()

        # Exponential barrier
        if self.exponential_bound != float("inf"):
            regularization += torch.sum(
                torch.exp(param_vec - self.exponential_bound)
                + torch.exp(-param_vec - self.exponential_bound)
            )

        # Dirichlet regularization
        if self.pseudocount > 0:
            # Get normalization value
            if experiment is None:
                norm = sum(
                    self.pseudocount / sum(expt.counts_per_round)
                    for expt in self.experiments
                ) / len(self.experiments)
            else:
                norm = self.pseudocount / sum(experiment.counts_per_round)

            # Calculate PDF
            log_pdf = torch.tensor(0.0, device=param_vec.device)
            for module in self.modules():
                if isinstance(module, PSAM):
                    log_pdf += module.get_dirichlet()
            regularization -= log_pdf * norm

        return regularization

    @override
    def forward(self, batch: Iterable[CountBatch]) -> Loss:
        """Calculates the multitask weighted Poisson NLL and regularization.

        Args:
            batch: Iterable of count tables to calculate the loss against.

        Returns:
            A NamedTuple with attributes `negloglik` (the Poisson NLL) and
            `regularization`, both as scalar tensors.
        """

        neglogliks: list[Tensor] = []
        try:
            # Calculate loss for each experiment
            for expt, sample in zip(self.experiments, batch, strict=True):
                device = expt.rounds[0].log_depth.device
                split_size = get_split_size(
                    self.max_embedding_size(),
                    (
                        len(sample.seqs)
                        if self.max_split is None
                        else min(self.max_split, len(sample.seqs))
                    ),
                    device,
                )

                # Split calculation into minibatches of split_size
                curr_nll = torch.tensor(0.0, device=device)
                sum_counts = torch.tensor(0.0, device=device)
                for seqs, target in zip(
                    torch.split(sample.seqs, split_size),
                    torch.split(sample.target, split_size),
                ):
                    seqs, target = seqs.to(device), target.to(device)
                    if self.full_loss:
                        loglik = (
                            (target * expt(seqs))
                            + (
                                target
                                * torch.log(target.sum(dim=1, keepdim=True))
                            )
                            - target
                            - torch.lgamma(target + 1)
                        )
                    else:
                        loglik = target * expt(seqs)
                    curr_nll -= torch.sum(loglik)
                    sum_counts += torch.sum(target)

                neglogliks.append(curr_nll / sum_counts)

        except ValueError as e:
            if str(e).startswith("zip"):
                raise ValueError(
                    "Length of experiments and batches may not match"
                ) from e
            raise e

        # Get scaling factors for each experiment
        weights = self.weights
        if self.equalize_contribution:
            with torch.inference_mode():
                sum_neglogliks = sum(nll.item() for nll in neglogliks)
                sum_weights = sum(weights)
                weights = [
                    (weight / sum_weights)
                    * (sum_neglogliks / cast(float, loss.item()))
                    for weight, loss in zip(weights, neglogliks)
                ]

        # Get regularization
        if self.dilute_regularization:
            regularization: Tensor | Literal[0] = self.regularization()
        else:
            regularization = sum(
                norm * self.regularization(expt)
                for norm, expt in zip(weights, self.experiments)
            )

        # Multiply losses by weights
        negloglik = sum(norm * nll for norm, nll in zip(weights, neglogliks))
        return Loss(cast(Tensor, negloglik), cast(Tensor, regularization))
