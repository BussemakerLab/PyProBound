"""Curve fitting to independent validation data."""

import abc
import copy
import os
from collections.abc import Callable, Iterable, Iterator, MutableMapping
from typing import Any, TypeVar

import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import Sampler
from typing_extensions import override

from . import __precision__
from .aggregate import Aggregate
from .base import Spec
from .cooperativity import Cooperativity
from .layers import Layer
from .loss import Loss, LossModule
from .mode import Mode
from .optimizer import Optimizer
from .plotting import gnbu_mod
from .rounds import BaseRound
from .table import CountBatch, CountTable
from .utils import avg_pool1d, get_split_size

T = TypeVar("T")


class BaseFit(LossModule[CountBatch], abc.ABC):
    """Base class for curve fitting to independent validation data."""

    scale: torch.nn.Parameter
    intercept: torch.nn.Parameter

    def __init__(
        self,
        rnd: BaseRound | Aggregate,
        dataset: CountTable,
        prediction: Callable[[Tensor], Tensor],
        observation: Callable[[Tensor], Tensor] = lambda x: x,
        update_construct: bool = False,
        train_posbias: bool = False,
        train_hill: bool = False,
        max_split: int | None = None,
        batch_size: int | None = None,
        checkpoint: str | os.PathLike[str] = "valmodel.pt",
        output: str | os.PathLike[str] = os.devnull,
        device: str | None = None,
        sampler: type[Sampler[CountBatch]] | None = None,
        optimizer: type[torch.optim.Optimizer] = torch.optim.LBFGS,
        optim_args: MutableMapping[str, Any] | None = None,
        sampler_args: MutableMapping[str, Any] | None = None,
        name: str = "",
    ) -> None:
        r"""Initializes the curve fitting.

        Args:
            rnd: A component containing an aggregate of different modes.
            dataset: A CountTable with 1 to 3 columns, with the first column
                taken as the target; if 2 columns are provided, the second
                column is taken as a symmetrical error; if 3 columns are
                provided, the second is taken as the lower error and the third
                is taken as the upper error.
            prediction: A callable applied to the log aggregate :math:`\log Z`.
            observation: A callable applied to the target :math:`y`.
            update_construct: Whether to reset experiment-specific parameters.
            train_posbias: Whether to retrain positional bias profiles
                :math:`\omega`.
            train_hill: Whether to train a Hill coefficient.
            max_split: Maximum number of sequences scored at a time
                (lower values reduce memory but increase computation time).
            batch_size: The number of sequences used to optimize the model at a
                time.
            checkpoint: The file where the model will be checkpointed to.
            output: The file where the optimization output will be written to.
            device: The device on which to perform optimization.
            sampler: The sampler used when creating the dataloader.
            optimizer: The optimizer used for optimization.
            optim_args: Parameters passed to the optimizer.
                (Defaults to `{"line_search_fn":"strong_wolfe"}` if available).
            sampler_args: Parameters passed to the sampler.
            name: A string used to describe the validation dataset.
        """
        super().__init__()

        # Get device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Instance attributes
        self.table = dataset
        self.prediction = prediction
        self.observation = observation
        self.train_posbias = train_posbias
        self.max_split = max_split
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.name = name

        # Set up round
        self.round = copy.deepcopy(rnd).to(self.device)
        self.round.eval()
        self.round.freeze()
        if isinstance(self.round, BaseRound):
            self.round.train_depth = False
            self.round.log_depth.zero_()
        for mod in self.round.modules():
            if isinstance(mod, Mode):
                mod.train_hill = train_hill

                if update_construct:
                    mod.update_read_length(
                        left_shift=dataset.input_shape - mod.input_shape,
                        max_len_shift=(
                            (dataset.max_read_length - dataset.min_read_length)
                            - (mod.max_input_length - mod.min_input_length)
                        ),
                        new_min_len=dataset.min_read_length,
                        new_max_len=dataset.max_read_length,
                    )

            elif isinstance(mod, Cooperativity):
                mod.train_hill = train_hill
                mod.train_posbias = False
                if update_construct:
                    for diag in mod.log_posbias:
                        diag.zero_()

            elif isinstance(mod, Layer):
                if update_construct:
                    for parameter in mod.parameters(recurse=False):
                        parameter.zero_()

        self.round.check_length_consistency()

        # Unfreeze all parameters that aren't in a Layer
        self.round.unfreeze("all")
        for mod in self.round.modules():
            if isinstance(mod, (Layer, Spec)):
                mod.freeze()

        # Set up optimizer
        self.optimizer = Optimizer(
            self,
            [self.table],
            epochs=50,
            patience=3,
            checkpoint=self.checkpoint,
            device=self.device.type,
            output=output,
            batch_size=batch_size,
            sampler=sampler,
            optimizer=optimizer,
            optim_args=optim_args,
            sampler_args=sampler_args,
        )

    @override
    def get_setup_string(self) -> str:
        return ""

    @override
    def components(self) -> Iterator[BaseRound] | Iterator[Aggregate]:
        yield self.round

    def log_aggregate(self, seqs: Tensor) -> Tensor:
        r"""Calculates the log aggregate :math:`\log Z_i`.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.

        Returns:
            The log aggregate tensor of shape :math:`(\text{minibatch},)`.
        """
        if isinstance(self.round, BaseRound):
            return self.round.log_aggregate(seqs)
        return self.round(seqs)

    @abc.abstractmethod
    def obs_pred(
        self, seqs: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        r"""Calculates the observed and predicted values used for the loss.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.
            target: A target tensor of shape
                :math:`(\text{minibatch},1-3)`

        Returns:
            A tuple of four tensors of shape :math:`(\text{minibatch},)`, being
            the transformed observed values, the transformed predicted values,
            the lower error values, and the upper error values.
        """

    def score(
        self, batch: CountBatch
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Wraps `obs_pred`, automatically managing devices."""
        split_size = get_split_size(
            self.max_embedding_size(),
            (
                len(batch.seqs)
                if self.max_split is None
                else min(self.max_split, len(batch.seqs))
            ),
            self.device,
        )
        observations: list[Tensor] = []
        predictions: list[Tensor] = []
        lower_error: list[Tensor] = []
        upper_error: list[Tensor] = []
        with torch.inference_mode():
            for seqs, target in zip(
                torch.split(batch.seqs, split_size),
                torch.split(batch.target, split_size),
            ):
                obs, pred, lower_err, upper_err = self.obs_pred(
                    seqs.to(self.device), target.to(self.device)
                )
                observations.append(obs.cpu())
                predictions.append(pred.cpu())
                if lower_err is not None:
                    lower_error.append(lower_err.cpu())
                if upper_err is not None:
                    upper_error.append(upper_err.cpu())
        return (
            torch.cat(observations),
            torch.cat(predictions),
            None if len(lower_error) == 0 else torch.cat(lower_error),
            None if len(upper_error) == 0 else torch.cat(upper_error),
        )

    @override
    def forward(self, batch: Iterable[CountBatch]) -> Loss:
        """Calculates the mean squared error loss."""
        curr_mse = torch.tensor(0.0, device=self.device)
        numel = torch.tensor(0.0, device=self.device)

        for sample in batch:
            split_size = get_split_size(
                self.max_embedding_size(),
                (
                    len(sample.seqs)
                    if self.max_split is None
                    else min(self.max_split, len(sample.seqs))
                ),
                self.device,
            )

            for seqs, target in zip(
                torch.split(sample.seqs, split_size),
                torch.split(sample.target, split_size),
            ):
                obs, pred, lower_err, upper_err = self.obs_pred(
                    seqs.to(self.device), target.to(self.device)
                )
                loss = torch.square(obs - pred)
                if lower_err is not None and upper_err is not None:
                    loss /= abs(upper_err - lower_err) / 2
                curr_mse += loss.sum()
                numel += loss.numel()

        return Loss(curr_mse / numel, torch.tensor(0.0))

    def _plot(
        self,
        obs: Tensor,
        pred: Tensor,
        err: Tensor | None = None,
        xlog: bool = True,
        ylog: bool = True,
        xlabel: str = "Predicted",
        ylabel: str = "Observed",
        kernel: int = 1,
        labels: list[str] | None = None,
        colors: list[str] | None = None,
    ) -> None:
        """Plots predicted validation values with error bars and binning."""
        pred, obs = pred.float(), obs.float()
        if kernel > 1:
            sorting = torch.argsort(pred, dim=0, descending=True)
            pred = avg_pool1d(pred[sorting], kernel=kernel)
            obs = avg_pool1d(obs[sorting], kernel=kernel)

        hexbin = len(pred) > 500 and err is None
        if hexbin:
            fig, axs = plt.subplots(
                ncols=2,
                figsize=(5.5, 5),
                gridspec_kw={"width_ratios": [5, 0.5]},
                constrained_layout=True,
            )
            axs[1].axis("off")
        else:
            fig, axs = plt.subplots(figsize=(5, 5), constrained_layout=True)

        if pred.min() <= 0 or obs.min() <= 0:
            xlog, ylog = False, False

        # Calculate linear regression
        pred_regress = np.log(pred) if xlog else pred.numpy()
        obs_regress = np.log(obs) if ylog else obs.numpy()
        slope, intercept, pearson, _, _ = scipy.stats.linregress(
            pred_regress, obs_regress
        )
        spearman = scipy.stats.spearmanr(obs, pred)[0]

        # Plot data
        scatter_label = f"$r_s$ ={spearman:.3f}, $r$ ={pearson:.3f}"
        if hexbin:
            binplot = axs[0].hexbin(
                pred,
                obs,
                cmap=gnbu_mod,
                label=scatter_label,
                bins="log",
                xscale="log" if xlog else "linear",
                yscale="log" if ylog else "linear",
            )
            fig.colorbar(binplot, ax=axs[1], fraction=1, pad=0)
        else:
            if err is None:
                axs.scatter(
                    pred, obs, label=scatter_label, alpha=0.5, color=colors
                )
            else:
                if colors is None:
                    axs.errorbar(
                        pred.numpy(),
                        obs.numpy(),
                        yerr=err.numpy(),
                        label=scatter_label,
                        alpha=0.5,
                        fmt="o",
                        capsize=7,
                    )
                else:
                    for i, (p, o, e, c) in enumerate(
                        zip(pred, obs, err.T, colors)
                    ):
                        axs.errorbar(
                            p,
                            o,
                            yerr=e.unsqueeze(-1),
                            label=scatter_label if i == 0 else None,
                            alpha=0.5,
                            color=c,
                            fmt="o",
                            capsize=7,
                        )
            if xlog:
                plt.xscale("log")
            if ylog:
                plt.yscale("log")

        # Set limits
        curr_ax = axs[0] if hexbin else axs
        min_range = min(curr_ax.get_xlim()[0], curr_ax.get_ylim()[0])
        max_range = max(curr_ax.get_xlim()[1], curr_ax.get_ylim()[1])
        curr_ax.set_xlim(min_range, max_range)
        curr_ax.set_ylim(min_range, max_range)

        # Plot linear regression
        lsrl_x = np.array(
            [
                min(obs.min().item(), pred.min().item()),
                max(obs.max().item(), pred.max().item()),
            ]
        )
        if xlog:
            lsrl_x = np.log(lsrl_x)
        lsrl_y = np.array([slope * i + intercept for i in lsrl_x])
        if xlog:
            lsrl_x = np.exp(lsrl_x)
        if ylog:
            lsrl_y = np.exp(lsrl_y)
        lsrl_label = (
            f"y = {slope:.3f}x"
            f" {'+' if intercept >= 0 else '-'} {abs(intercept):.3f}"
        )
        curr_ax.plot(lsrl_x, lsrl_y, "k--", label=lsrl_label)

        # Plot text
        title = self.name
        if kernel > 1:
            title += f" ({len(pred):,} bins of n={kernel})"
        else:
            title += f" ({len(pred):,} probes)"
        if labels is not None:
            for txt, x, y in zip(labels, pred, obs):
                curr_ax.annotate(txt, (x, y))
        curr_ax.set_title(title)
        curr_ax.set_xlabel(xlabel)
        curr_ax.set_ylabel(ylabel)
        curr_ax.legend()

    def _fit(self, log: bool = False) -> None:
        """Fits experiment-specific parameters to the validation data."""
        multiples = [0.1, 0.316, 1, 3.162, 10]
        best_loss = torch.tensor(float("inf"))
        original_state_dict = copy.deepcopy(self.state_dict())
        best_state_dict = copy.deepcopy(self.state_dict())

        for scale_m in multiples:
            for intercept_m in multiples:
                curr_state_dict = copy.deepcopy(original_state_dict)
                if log:
                    curr_state_dict["scale"] = (
                        np.log(scale_m) + original_state_dict["scale"]
                    )
                    curr_state_dict["intercept"] = (
                        np.log(intercept_m) + original_state_dict["intercept"]
                    )
                else:
                    curr_state_dict["scale"] = (
                        scale_m * original_state_dict["scale"]
                    )
                    curr_state_dict["intercept"] = (
                        intercept_m * original_state_dict["intercept"]
                    )
                self.optimizer.model.load_state_dict(curr_state_dict)
                loss = self.optimizer.train_simultaneous()
                if loss < best_loss:
                    best_loss = loss
                    best_state_dict = copy.deepcopy(
                        self.optimizer.model.state_dict()
                    )

        self.load_state_dict(best_state_dict)
        self.optimizer.save(self.checkpoint)

        # Unfreeze experiment-specific parameters
        if self.train_posbias:
            for mod in self.round.modules():
                if isinstance(mod, Layer):
                    for parameter in mod.parameters(recurse=False):
                        parameter.requires_grad_()
                elif isinstance(mod, Cooperativity):
                    if mod.train_posbias:
                        mod.log_posbias.requires_grad_()

            self.optimizer.train_simultaneous()
            self.optimizer.save(self.checkpoint)


class Fit(BaseFit):
    r"""Curve fitting to independent validation data in linear space.

    .. math::
        \text{observation} (y) \sim m \times \text{prediction} (\log Z) + b

    Attributes:
        scale (Tensor): The scaling factor :math:`m` (1 if not `train_offset`).
        intercept (Tensor): The intercept :math:`b` (0 if not `train_offset`).
    """

    def __init__(
        self,
        rnd: BaseRound | Aggregate,
        dataset: CountTable,
        prediction: Callable[[Tensor], Tensor],
        observation: Callable[[Tensor], Tensor] = lambda x: x,
        update_construct: bool = False,
        train_offset: bool = False,
        train_posbias: bool = False,
        train_hill: bool = False,
        max_split: int | None = None,
        batch_size: int | None = None,
        checkpoint: str | os.PathLike[str] = "valmodel.pt",
        output: str | os.PathLike[str] = os.devnull,
        device: str | None = None,
        sampler: type[Sampler[CountBatch]] | None = None,
        optimizer: type[torch.optim.Optimizer] = torch.optim.LBFGS,
        optim_args: MutableMapping[str, Any] | None = None,
        sampler_args: MutableMapping[str, Any] | None = None,
        name: str = "",
    ) -> None:
        r"""Initializes the curve fitting.

        Args:
            rnd: A component containing an aggregate of different modes.
            dataset: A CountTable with 1 to 3 columns, with the first column
                taken as the target; if 2 columns are provided, the second
                column is taken as a symmetrical error; if 3 columns are
                provided, the second is taken as the lower error and the third
                is taken as the upper error.
            prediction: A callable applied to the log aggregate :math:`\log Z`.
            observation: A callable applied to the target :math:`y`.
            update_construct: Whether to reset experiment-specific parameters.
            train_offset: Whether to train scaling and intercept parameters.
            train_posbias: Whether to retrain positional bias profiles
                :math:`\omega`.
            train_hill: Whether to train a Hill coefficient.
            max_split: Maximum number of sequences scored at a time
                (lower values reduce memory but increase computation time).
            batch_size: The number of sequences used to optimize the model at a
                time.
            checkpoint: The file where the model will be checkpointed to.
            output: The file where the optimization output will be written to.
            device: The device on which to perform optimization.
            sampler: The sampler used when creating the dataloader.
            optimizer: The optimizer used for optimization.
            optim_args: Parameters passed to the optimizer.
                (Defaults to `{"line_search_fn":"strong_wolfe"}` if available).
            sampler_args: Parameters passed to the sampler.
            name: A string used to describe the validation dataset.
        """
        super().__init__(
            rnd,
            dataset,
            prediction=prediction,
            observation=observation,
            update_construct=update_construct,
            train_posbias=train_posbias,
            train_hill=train_hill,
            max_split=max_split,
            batch_size=batch_size,
            checkpoint=checkpoint,
            output=output,
            device=device,
            sampler=sampler,
            optimizer=optimizer,
            optim_args=optim_args,
            sampler_args=sampler_args,
            name=name,
        )
        scale = 1.0
        intercept = 0.0
        if train_offset:
            scale = (self.table.target.max() - self.table.target.min()).item()
            intercept = self.table.target.min().item()
        self.scale = torch.nn.Parameter(
            torch.tensor(scale, dtype=__precision__),
            requires_grad=train_offset,
        )
        self.intercept = torch.nn.Parameter(
            torch.tensor(intercept, dtype=__precision__),
            requires_grad=train_offset,
        )

    @override
    def obs_pred(
        self, seqs: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        r"""Calculates the observed and predicted values used for the loss.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.
            target: A target tensor of shape
                :math:`(\text{minibatch},1-3)`

        Returns:
            A tuple of four tensors of shape :math:`(\text{minibatch},)`, being
            :math:`\text{obs}(y)`, :math:`m \times \text{pred}(\log Z) + b`,
            :math:`\text{obs}(y - \text{lower error})`,
            and :math:`\text{obs}(y + \text{lower error})`.
        """
        obs = self.observation(target[:, 0])
        lower_err = None
        upper_err = None
        if target.shape[1] > 1:
            lower_err = abs(self.observation(target[:, 0] - target[:, 1]))
            upper_err = abs(self.observation(target[:, 0] + target[:, -1]))
            lower_err, upper_err = torch.sort(
                torch.stack((lower_err, upper_err)), dim=0
            ).values

        pred = self.intercept + (
            self.scale * self.prediction(self.log_aggregate(seqs))
        )
        return obs, pred, lower_err, upper_err

    def plot(
        self,
        xlabel: str = "Predicted",
        ylabel: str = "Observed",
        kernel: int = 1,
        xlog: bool = True,
        ylog: bool = True,
        labels: list[str] | None = None,
        colors: list[str] | None = None,
    ) -> None:
        """Plots predicted validation values with error bars and binning.

        Args:
            xlabel: The x-axis label.
            ylabel: The y-axis label.
            kernel: The bin for average pooling of prediction-sorted sequences.
            xlog: Whether to plot the x-axis in logarithmic scale.
            ylog: Whether to plot the y-axis in logarithmic scale.
            labels: The label for each data point drawn on the plot.
            colors: The color for each data point drawn on the plot.
        """
        with torch.inference_mode():
            obs, pred, lower_err, upper_err = self.score(self.table)
        err = None
        if lower_err is not None and upper_err is not None:
            err = torch.stack((obs - lower_err, upper_err - obs), dim=0)
        self._plot(
            obs=obs,
            pred=pred,
            err=err,
            xlog=xlog,
            ylog=ylog,
            xlabel=xlabel,
            ylabel=ylabel,
            kernel=kernel,
            labels=labels,
            colors=colors,
        )

    def fit(self) -> None:
        """Fits experiment-specific parameters to the validation data."""
        super()._fit(log=False)


class LogFit(BaseFit):
    r"""Curve fitting to independent validation data in logarithmic space.

    .. math::
        \log \left( \text{observation} (y) \right) \sim \log \left(
        \exp(m) \times \exp \left( \text{prediction} (\log Z) \right) + \exp(b)
        \right)

    Attributes:
        scale (Tensor): The scaling factor :math:`m` (0 if not `train_offset`).
        intercept (Tensor): The intercept :math:`b` (-∞ if not `train_offset`).
    """

    def __init__(
        self,
        rnd: BaseRound | Aggregate,
        dataset: CountTable,
        prediction: Callable[[Tensor], Tensor],
        observation: Callable[[Tensor], Tensor] = lambda x: x,
        update_construct: bool = False,
        train_offset: bool = False,
        train_posbias: bool = False,
        train_hill: bool = False,
        max_split: int | None = None,
        batch_size: int | None = None,
        checkpoint: str | os.PathLike[str] = "valmodel.pt",
        output: str | os.PathLike[str] = os.devnull,
        device: str | None = None,
        sampler: type[Sampler[CountBatch]] | None = None,
        optimizer: type[torch.optim.Optimizer] = torch.optim.LBFGS,
        optim_args: MutableMapping[str, Any] | None = None,
        sampler_args: MutableMapping[str, Any] | None = None,
        name: str = "",
    ) -> None:
        r"""Initializes the curve fitting.

        Args:
            rnd: A component containing an aggregate of different modes.
            dataset: A CountTable with 1 to 3 columns, with the first column
                taken as the target; if 2 columns are provided, the second
                column is taken as a symmetrical error; if 3 columns are
                provided, the second is taken as the lower error and the third
                is taken as the upper error.
            prediction: A callable applied to the log aggregate :math:`\log Z`.
            observation: A callable applied to the target :math:`y`.
            update_construct: Whether to reset experiment-specific parameters.
            train_offset: Whether to train scaling and intercept parameters.
            train_posbias: Whether to retrain positional bias profiles
                :math:`\omega`.
            train_hill: Whether to train a Hill coefficient.
            max_split: Maximum number of sequences scored at a time
                (lower values reduce memory but increase computation time).
            batch_size: The number of sequences used to optimize the model at a
                time.
            checkpoint: The file where the model will be checkpointed to.
            output: The file where the optimization output will be written to.
            device: The device on which to perform optimization.
            sampler: The sampler used when creating the dataloader.
            optimizer: The optimizer used for optimization.
            optim_args: Parameters passed to the optimizer.
                (Defaults to `{"line_search_fn":"strong_wolfe"}` if available).
            sampler_args: Parameters passed to the sampler.
            name: A string used to describe the validation dataset.
        """
        super().__init__(
            rnd,
            dataset,
            prediction=prediction,
            observation=observation,
            update_construct=update_construct,
            train_posbias=train_posbias,
            train_hill=train_hill,
            max_split=max_split,
            batch_size=batch_size,
            checkpoint=checkpoint,
            output=output,
            device=device,
            sampler=sampler,
            optimizer=optimizer,
            optim_args=optim_args,
            sampler_args=sampler_args,
            name=name,
        )
        scale = 0.0
        intercept = float("-inf")
        if train_offset:
            scale = (
                (self.table.target.max() - self.table.target.min())
                .log()
                .item()
            )
            intercept = self.table.target.min().log().item()
        self.scale = torch.nn.Parameter(
            torch.tensor(scale, dtype=__precision__),
            requires_grad=train_offset,
        )
        self.intercept = torch.nn.Parameter(
            torch.tensor(intercept, dtype=__precision__),
            requires_grad=train_offset,
        )

    @override
    def obs_pred(
        self, seqs: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        r"""Calculates the observed and predicted values used for the loss.

        Args:
            seqs: A sequence tensor of shape
                :math:`(\text{minibatch},\text{length})` or
                :math:`(\text{minibatch},\text{in_channels},\text{length})`.
            target: A target tensor of shape
                :math:`(\text{minibatch},1-3)`

        Returns:
            A tuple of four tensors of shape :math:`(\text{minibatch},)`, being
            :math:`\log\text{obs}(y)`,
            :math:`\log (\exp(m + \text{prediction} (\log Z) ) + \exp(b))`,
            :math:`\log\text{obs}(y - \text{lower error})`,
            and :math:`\log\text{obs}(y + \text{lower error})`.
        """
        obs = self.observation(target[:, 0])
        lower_err = None
        upper_err = None
        if target.shape[1] > 1:
            lower_err = abs(
                self.observation(target[:, 0] - target[:, 1])
            ).log()
            upper_err = abs(
                self.observation(target[:, 0] + target[:, -1])
            ).log()
            lower_err, upper_err = torch.sort(
                torch.stack((lower_err, upper_err)), dim=0
            ).values
        pred = self.scale + self.prediction(self.log_aggregate(seqs))
        if not torch.isneginf(self.intercept):
            pred = torch.logaddexp(self.intercept, pred)
        return obs.log(), pred, lower_err, upper_err

    def plot(
        self,
        xlabel: str = "Predicted",
        ylabel: str = "Observed",
        kernel: int = 1,
        xlog: bool = True,
        ylog: bool = True,
        labels: list[str] | None = None,
        colors: list[str] | None = None,
    ) -> None:
        """Plots predicted validation values with error bars and binning.

        Args:
            xlabel: The x-axis label.
            ylabel: The y-axis label.
            kernel: The bin for average pooling of prediction-sorted sequences.
            xlog: Whether to plot the x-axis in logarithmic scale.
            ylog: Whether to plot the y-axis in logarithmic scale.
            labels: The label for each data point drawn on the plot.
            colors: The color for each data point drawn on the plot.
        """
        with torch.inference_mode():
            obs, pred, lower_err, upper_err = self.score(self.table)
            obs = obs.exp()
            pred = pred.exp()
        err = None
        if lower_err is not None and upper_err is not None:
            lower_err = lower_err.exp()
            upper_err = upper_err.exp()
            err = torch.stack((obs - lower_err, upper_err - obs), dim=0)
        self._plot(
            obs=obs,
            pred=pred,
            err=err,
            xlog=xlog,
            ylog=ylog,
            xlabel=xlabel,
            ylabel=ylabel,
            kernel=kernel,
            labels=labels,
            colors=colors,
        )

    def fit(self) -> None:
        """Fits experiment-specific parameters to the validation data."""
        super()._fit(log=True)
