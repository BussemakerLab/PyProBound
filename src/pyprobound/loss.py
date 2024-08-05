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
from .base import Component, Transform
from .containers import TModuleList
from .experiment import Experiment
from .layers import PSAM
from .table import Batch, CountBatch
from .utils import get_split_size

T = TypeVar("T", bound=Batch)


class Loss(NamedTuple):
    """A loss value, interpreted as the sum of both elements.

    Attributes:
        negloglik: The negative log likelihood.
        regularization: The regularization value.
    """

    negloglik: Tensor
    regularization: Tensor


class BaseLoss(Component, Generic[T]):
    """Transform that calculates loss.

    Attributes:
        transforms (TModuleList[Transform]): The components to be jointly
            optimized.
    """

    def __init__(
        self,
        components: Iterable[Transform],
        weights: Iterable[float] | None = None,
        lambda_l2: float = 1e-6,
        lambda_l1: float = 0,
        exponential_bound: float = 40,
        dilute_regularization: bool = False,
        exclude_regularization: Iterable[str] = tuple(),
        equalize_contribution: bool = False,
        max_split: int | None = None,
    ) -> None:
        """Initializes the multitask loss from an iterable of components.

        Args:
            transforms: The components to be jointly optimized.
            weights: Multiplier to the NLL of each corresponding component.
            lambda_l2: L2 regularization hyperparameter.
            lambda_l1: L1 regularization hyperparameter.
            exponential_bound: Value of exponential barrier.
            dilute_regularization: Whether to keep hyperparameters fixed as the
                number of components increases.
            exclude_regularization: Keywords used to exclude parameters from
                regularization.
            equalize_contribution: Whether to update the weights so that the
                rescaled losses are constant relative to each other.
            max_split: Maximum number of sequences scored at a time.
        """
        super().__init__(name="")

        # Store loss attributes
        self.transforms = TModuleList(components)
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.exponential_bound = exponential_bound
        self.dilute_regularization = dilute_regularization
        self.exclude_regularization = tuple(exclude_regularization)
        self.equalize_contribution = equalize_contribution
        self.max_split = max_split

        # Update names
        for transform_idx, transform in enumerate(self.transforms):
            if transform.name == "":
                transform.name = str(transform_idx)

        # Store scaling factor for loss of each experiment
        if weights is None:
            weights = [1 / len(self.transforms)] * len(self.transforms)
        else:
            weights = list(weights)
        self.weights = weights
        if len(self.weights) != len(self.transforms):
            raise ValueError(
                f"Length of weights {len(self.weights)} does not match"
                f" number of components {len(self.transforms)}"
            )

    @override
    def components(self) -> Iterator[Transform]:
        return iter(self.transforms)

    @abc.abstractmethod
    def negloglik(
        self, transform: Transform, tensors: Iterable[Tensor]
    ) -> tuple[Tensor, Tensor | float]:
        """Calculates the negative log-likelihood plus a normalization factor.

        Args:
            transform: The component to be scored.
            tensors: The tensors in a batch.

        Returns:
            A tuple of scalar tensors (negloglik, norm), where negloglik is the
            negative log-likelihood of the batch and norm is a scaling factor.
            A running sum of each is kept, and the loss is the ratio.
        """

    def get_setup_string(self) -> str:
        """A description used when printing the output of an optimizer."""
        return "\n".join(
            (
                "### Regularization:",
                f"\t L1 Lambda: {self.lambda_l1}",
                f"\t L2 Lambda: {self.lambda_l2}",
                f"\t Exponential Bound: {self.exponential_bound}",
                f"\t Excluded Reg.: {self.exclude_regularization}",
                f"\t Eq. Contribution: {self.equalize_contribution}",
                f"\t Weights: {self.weights}",
            )
        )

    def regularization(self, component: Component) -> Tensor:
        """Calculates parameter regularization.

        Args:
            component: The component containing parameters to be regularized.

        Returns:
            The regularization value as a scalar tensor.
        """
        # Get flattened parameter vector
        param_list = []
        for name, param in component.named_parameters():
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

        return regularization

    @override
    def forward(self, batches: Iterable[T]) -> Loss:
        """Calculates the multitask weighted loss and regularization.

        Args:
            batches: Iterable of batches to calculate the loss against.

        Returns:
            A NamedTuple with attributes `negloglik` and `regularization`,
            both as scalar tensors.
        """

        for param in self.parameters():
            device = param.device
            break

        neglogliks: list[Tensor] = []
        try:
            # Calculate loss for each component
            for transform, batch in zip(self.transforms, batches, strict=True):
                batchlen = len(next(batch.tensors()))
                split_size = get_split_size(
                    self.max_embedding_size(),
                    (
                        batchlen
                        if self.max_split is None
                        else min(self.max_split, batchlen)
                    ),
                    device,
                )

                # Split calculation into minibatches of split_size
                curr_nll = torch.tensor(0.0, device=device)
                curr_norm = torch.tensor(0.0, device=device)
                for elements in zip(
                    *(torch.split(i, split_size) for i in batch.tensors())
                ):
                    elements = tuple(i.to(device) for i in elements)
                    negloglik, norm = self.negloglik(transform, elements)
                    curr_nll += negloglik
                    curr_norm += norm
                neglogliks.append(curr_nll / curr_norm)

        except ValueError as e:
            if str(e).startswith("zip"):
                raise ValueError(
                    "Length of components and batches may not match"
                ) from e
            raise e

        # Get scaling factors for each component
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
            regularization: Tensor | Literal[0] = self.regularization(self)
        else:
            regularization = sum(
                norm * self.regularization(transform)
                for norm, transform in zip(weights, self.transforms)
            )

        # Multiply losses by weights
        final_nll = sum(norm * nll for norm, nll in zip(weights, neglogliks))
        return Loss(cast(Tensor, final_nll), cast(Tensor, regularization))

    @override
    def __call__(self, batches: Iterable[T]) -> Loss:
        """See https://github.com/pytorch/pytorch/issues/45414."""
        return cast(Loss, super().__call__(batches))


class MultiExperimentLoss(BaseLoss[CountBatch]):
    """Multitask optimization of multiple count tables with a Poisson loss.

    Attributes:
        transforms (TModuleList[Experiment]): The experiments to be jointly
            optimized.
    """

    def __init__(
        self,
        components: Iterable[Experiment],
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
            components: The experiments to be jointly optimized.
            weights: Multiplier to the NLL of each corresponding experiment.
            lambda_l2: L2 regularization hyperparameter.
            lambda_l1: L1 regularization hyperparameter.
            pseudocount: Scaling factor for Dirichlet-inspired regularization.
            exponential_bound: Value of exponential barrier.
            full_loss: Whether to compute the constant terms of the NLL.
            dilute_regularization: Whether to keep hyperparameters fixed as the
                number of experiments increases.
            exclude_regularization: Keywords used to exclude parameters from
                regularization.
            equalize_contribution: Whether to update the weights so that the
                rescaled losses are constant relative to each other.
            max_split: Maximum number of sequences scored at a time.
        """
        super().__init__(
            components=components,
            weights=weights,
            lambda_l2=lambda_l2,
            lambda_l1=lambda_l1,
            exponential_bound=exponential_bound,
            dilute_regularization=dilute_regularization,
            exclude_regularization=exclude_regularization,
            equalize_contribution=equalize_contribution,
            max_split=max_split,
        )
        self.full_loss = full_loss
        self.pseudocount = pseudocount

        # Maintain compatability with older fits where `transforms` attribute
        # was called `experiments` while avoiding registering as a submodule
        object.__setattr__(  # Equiv. to `self.experiments = self.transforms`
            self, "experiments", self.transforms
        )

    @override
    def get_setup_string(self) -> str:
        out = [
            super().get_setup_string()
            + f"\n\t Pseudocount: {self.pseudocount}"
        ]

        out.append("\n### Experiments:")
        for expt in self.transforms:
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

        out.append("\n### Binding Transforms:")
        for binding_idx, (binding, optim) in enumerate(
            self.optim_procedure().items()
        ):
            out.extend([f"\t Mode {binding_idx}: {binding}"])
            for ancestors in optim.ancestry:
                out.append(f"\t\t{ancestors[-2]}")

        return "\n".join(out)

    @override
    def regularization(self, component: Component) -> Tensor:
        """Calculates parameter regularization.

        Args:
            transform: The component containing parameters to be regularized.

        Returns:
            The regularization value as a scalar tensor.
        """
        regularization = super().regularization(component)

        # Dirichlet regularization
        if self.pseudocount > 0:
            # Get normalization value
            if component is self:
                norm = sum(
                    self.pseudocount / sum(expt.counts_per_round)
                    for expt in self.transforms
                ) / len(self.transforms)
            else:
                norm = self.pseudocount / sum(component.counts_per_round)

            # Calculate PDF
            log_pdf = torch.tensor(0.0, device=regularization.device)
            for module in self.modules():
                if isinstance(module, PSAM):
                    log_pdf += module.get_dirichlet()
            regularization -= log_pdf * norm

        return regularization

    @override
    def negloglik(
        self, transform: Transform, tensors: Iterable[Tensor]
    ) -> tuple[Tensor, Tensor]:
        seqs, target = tensors
        if self.full_loss:
            loglik = (
                (target * transform(seqs))
                + (target * torch.log(target.sum(dim=1, keepdim=True)))
                - target
                - torch.lgamma(target + 1)
            )
        else:
            loglik = target * transform(seqs)
        return -torch.sum(loglik), torch.sum(target)


class MultiRoundMSLELoss(BaseLoss[CountBatch]):
    """Multitask optimization of intensity experiments with a MSLE loss.

    Attributes:
        transforms (TModuleList[BaseRound]): The rounds to be jointly
            optimized.
    """

    @override
    def negloglik(
        self, transform: Transform, tensors: Iterable[Tensor]
    ) -> tuple[Tensor, int]:
        seqs, target = tensors
        assert target.ndim == 2 and target.shape[-1] == 1
        assert transform.reference_round is None
        return (target[:, 0].log() - transform(seqs)).square().sum(), len(
            target
        )
