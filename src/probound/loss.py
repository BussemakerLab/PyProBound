"""Multi-experiment loss"""
import abc
from collections.abc import Iterable, Iterator
from typing import Generic, NamedTuple, TypeVar, cast

import torch
from torch import Tensor
from typing_extensions import override

from . import __precision__
from .base import Component, Spec, Transform
from .binding import BindingMode
from .containers import Buffer, TModuleList
from .cooperativity import BindingCooperativity
from .experiment import Experiment
from .psam import PSAM
from .rounds import _ARound
from .table import Batch
from .utils import get_ordinal, get_split_size

T = TypeVar("T")


class Loss(NamedTuple):
    negloglik: Tensor
    regularization: Tensor


class LossModule(Component, Generic[T]):
    """Component that calculates loss

    See https://github.com/pytorch/pytorch/issues/45414"""

    @override
    @abc.abstractmethod
    def forward(self, batch: Iterable[T]) -> Loss:
        """Loss"""

    @override
    @abc.abstractmethod
    def components(self) -> Iterator[Transform]:
        ...

    @override
    def __call__(self, batch: Iterable[T]) -> Loss:
        return cast(Loss, super().__call__(batch))

    @abc.abstractmethod
    def get_setup_string(self) -> str:
        ...


class MultiExperimentLoss(LossModule[Batch]):
    """Sequence of experiments for multitask optimization"""

    def __init__(
        self,
        experiments: Iterable[Experiment],
        weights: Iterable[float] | None = None,
        lambda_l2: float = 1e-6,
        lambda_l1: float = 0,
        pseudocount: float = 0,
        exponential_bound: float = 40,
        full_loss: bool = False,
        exclude_regularization: Iterable[str] = frozenset(),
        equalize_contribution: bool = False,
        max_split: int | None = None,
    ) -> None:
        super().__init__(name="")

        # store experiment attribute
        self.experiments: TModuleList[Experiment] = TModuleList(experiments)

        # store scaling factor for loss of each experiment
        if weights is None:
            weights = [1 / len(self.experiments)] * len(self.experiments)
        self.weights: Tensor = Buffer(
            torch.tensor(weights, dtype=__precision__)
        )
        if len(self.weights) != len(self.experiments):
            raise ValueError(
                f"Length of weights {len(self.weights)} does not match"
                f" number of experiments {len(self.experiments)}"
            )

        # store loss attributes
        self.exclude_regularization = exclude_regularization
        self.full_loss = full_loss
        self.equalize_contribution = equalize_contribution
        self.lambda_l2: Tensor = Buffer(
            torch.tensor(lambda_l2, dtype=__precision__)
        )
        self.lambda_l1: Tensor = Buffer(
            torch.tensor(lambda_l1, dtype=__precision__)
        )
        self.pseudocount: Tensor = Buffer(
            torch.tensor(pseudocount, dtype=__precision__)
        )
        self.exponential_bound: Tensor = Buffer(
            torch.tensor(exponential_bound, dtype=__precision__)
        )
        self.max_split = max_split

        # fill in spec names for each type
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

        # add ancestry information to component names
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
                if isinstance(mod, BindingMode):
                    mod.name = f"{expt}→" + "-".join(str(i) for i in mod.key())
                    for layer_idx, layer in enumerate(mod.layers):
                        layer.name = (
                            f"{mod}→Layer{layer_idx}:{layer.layer_spec}←"
                        )
                elif isinstance(mod, BindingCooperativity):
                    if mod.spacing.name == "":
                        mod.spacing.name = (
                            "-".join(
                                str(i) for i in mod.spacing.binding_mode_key_0
                            )
                            + "::"
                            + "-".join(
                                str(i) for i in mod.spacing.binding_mode_key_1
                            )
                        )
                    mod.name = f"{expt}→{mod.spacing.name}"

        # check that all names are unique
        specs = [mod for mod in self.modules() if isinstance(mod, Spec)]
        if len(specs) != len(set(str(spec) for spec in specs)):
            raise ValueError("Binding component names are not unique")
        if len(self.experiments) != len(
            set(expt.name for expt in self.experiments)
        ):
            raise ValueError("Experiment names are not unique")
        rnds = [mod for mod in self.modules() if isinstance(mod, _ARound)]
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
        for exp in self.experiments:
            round_format = [
                str(rnd) if rnd in exp.observed_rounds else f"({str(rnd)})"
                for rnd in exp.rounds
            ]
            out.extend(
                [
                    f"\tExperiment: {str(exp)}",
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

    def regularization(self, experiment: Experiment) -> Tensor:
        """Calculate parameter regularization"""

        # get flattened parameter vector
        param_vec = torch.cat(
            [
                param.flatten()
                for name, param in experiment.named_parameters()
                if (torch.all(torch.isfinite(param)))
                and not any(
                    exclude in name for exclude in self.exclude_regularization
                )
            ]
        )
        regularization = torch.tensor(0.0, device=param_vec.device)

        # add L2 regularization
        if self.lambda_l2 > 0:
            regularization += self.lambda_l2 * param_vec.square().sum()

        # add L1 regularization
        if self.lambda_l1 > 0:
            regularization += self.lambda_l1 * param_vec.abs().sum()

        # add Dirichlet regularization
        if self.pseudocount > 0:
            log_pdf = torch.tensor(0.0, device=param_vec.device)
            for module in self.modules():
                if isinstance(module, PSAM):
                    log_pdf += module.get_dirichlet()

            regularization -= log_pdf * (
                self.pseudocount / torch.sum(experiment.counts_per_round)
            )

        # add exponential barrier
        if not torch.isposinf(self.exponential_bound):
            regularization += torch.sum(
                torch.exp(param_vec - self.exponential_bound)
                + torch.exp(-param_vec - self.exponential_bound)
            )

        return regularization

    @override
    def forward(self, batch: Iterable[Batch]) -> Loss:
        """Calculate Poisson Negative Log-Likelihood and regularization"""

        neglogliks: list[Tensor] = []
        regularizations: list[Tensor] = []

        try:
            for exp, sample in zip(self.experiments, batch, strict=True):
                # pylint: disable-next=protected-access
                device = exp._counts_per_round.device
                split_size = get_split_size(
                    self.max_embedding_size(),
                    len(sample.seqs)
                    if self.max_split is None
                    else min(self.max_split, len(sample.seqs)),
                    device,
                )

                curr_nll = torch.tensor(0.0, device=device)
                sum_counts = torch.tensor(0.0, device=device)
                for seqs, target in zip(
                    torch.split(sample.seqs, split_size),
                    torch.split(sample.target, split_size),
                ):
                    seqs, target = seqs.to(device), target.to(device)
                    if self.full_loss:
                        loglik = (
                            (target * exp.log_prediction(seqs, target))
                            - target
                            - cast(Tensor, torch.special.gammaln(target + 1))
                        )
                    else:
                        loglik = target * exp(seqs)
                    curr_nll -= torch.sum(loglik)
                    sum_counts += torch.sum(target)

                neglogliks.append(curr_nll / sum_counts)
                regularizations.append(self.regularization(exp))

        except ValueError as e:
            raise ValueError(
                "Length of experiments and batches may not match?"
            ) from e

        weights = cast(list[float], self.weights.tolist())
        if self.equalize_contribution:
            with torch.inference_mode():
                sum_neglogliks = cast(
                    float, torch.sum(torch.cat(neglogliks, dim=0)).item()
                )
                sum_weights = sum(weights)
                weights = [
                    (weight / sum_weights)
                    * (sum_neglogliks / cast(float, loss.item()))
                    for weight, loss in zip(weights, neglogliks)
                ]

        negloglik = sum(norm * nll for norm, nll in zip(weights, neglogliks))
        regularization = sum(
            norm * reg for norm, reg in zip(weights, regularizations)
        )
        return Loss(cast(Tensor, negloglik), cast(Tensor, regularization))
