"""Nonlinear curve fitting"""
import abc
import copy
import os
from collections.abc import Callable, Iterable, Iterator

import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from typing_extensions import override

from . import __precision__
from .aggregate import Aggregate
from .base import Spec
from .binding import BindingMode
from .conv1d import Conv1d
from .cooperativity import BindingCooperativity
from .loss import Loss, LossModule
from .optimizer import Optimizer
from .plotting import gnbu_mod, save_image
from .rounds import _ARound
from .table import Batch, CountTable
from .utils import avg_pool1d, get_split_size


class _Fit(LossModule[Batch], abc.ABC):
    """Abstract base class for fitting to independent validation data"""

    scale: torch.nn.Parameter
    intercept: torch.nn.Parameter

    def __init__(
        self,
        rnd: _ARound | Aggregate,
        dataset: CountTable,
        prediction: Callable[[Tensor], Tensor],
        observation: Callable[[Tensor], Tensor] = lambda x: x,
        update_construct: bool = False,
        train_omega: bool = False,
        train_theta: bool = False,
        train_hill: bool = False,
        max_split: int | None = None,
        checkpoint: str | os.PathLike[str] = "valmodel.pt",
        output: str | os.PathLike[str] = os.devnull,
        device: str | None = None,
        name: str = "",
    ) -> None:
        super().__init__()

        # get device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # instance attributes
        self.table = dataset
        self.prediction = prediction
        self.observation = observation
        self.train_omega = train_omega
        self.train_theta = train_theta
        self.max_split = max_split
        self.checkpoint = checkpoint
        self.output = output
        self.device = torch.device(device)
        self.name = name

        # set up round
        self.round = copy.deepcopy(rnd).to(self.device)
        self.round.eval()
        self.round.freeze()
        if isinstance(self.round, _ARound):
            self.round.train_eta = False
            self.round.log_eta.zero_()
        for mod in self.round.modules():
            if isinstance(mod, BindingMode):
                mod.train_hill = train_hill
                for layer in mod.layers:
                    if isinstance(layer, Conv1d):
                        layer.train_omega = False
                        layer.train_theta = False

                        if update_construct:
                            layer.omega.zero_()
                            layer.theta.zero_()

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

            elif isinstance(mod, BindingCooperativity):
                mod.train_hill = train_hill
                mod.train_omega = False
                if update_construct:
                    mod.omega.zero_()

        # unfreeze experiment-specific parameters only
        self.round.unfreeze("all")
        for mod in self.round.modules():
            if isinstance(mod, Spec):
                mod.freeze()

        self.round.check_length_consistency()

    @override
    def components(self) -> Iterator[_ARound] | Iterator[Aggregate]:
        yield self.round

    @override
    def get_setup_string(self) -> str:
        return ""

    def log_aggregate(self, seqs: Tensor) -> Tensor:
        if isinstance(self.round, _ARound):
            return self.round.log_aggregate(seqs)
        return self.round(seqs)

    @abc.abstractmethod
    def obs_pred(
        self, seqs: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        pass

    def score(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        split_size = get_split_size(
            self.max_embedding_size(),
            len(batch.seqs)
            if self.max_split is None
            else min(self.max_split, len(batch.seqs)),
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
    def forward(self, batch: Iterable[Batch]) -> Loss:
        """Calculate mean squared error loss"""
        curr_mse = torch.tensor(0.0, device=self.device)
        numel = torch.tensor(0.0, device=self.device)

        for sample in batch:
            split_size = get_split_size(
                self.max_embedding_size(),
                len(sample.seqs)
                if self.max_split is None
                else min(self.max_split, len(sample.seqs)),
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
        save: bool | str = False,
    ) -> None:
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

        # calculate linear regression
        pred_regress = np.log(pred) if xlog else pred.numpy()
        obs_regress = np.log(obs) if ylog else obs.numpy()
        slope, intercept, pearson, _, _ = scipy.stats.linregress(
            pred_regress, obs_regress
        )
        spearman = scipy.stats.spearmanr(obs, pred)[0]

        # plot data
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
                        pred,
                        obs,
                        yerr=err,
                        label=scatter_label,
                        alpha=0.5,
                        fmt="o",
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
                        )
            if xlog:
                plt.xscale("log")
            if ylog:
                plt.yscale("log")

        # set limits
        curr_ax = axs[0] if hexbin else axs
        min_range = min(curr_ax.get_xlim()[0], curr_ax.get_ylim()[0])
        max_range = max(curr_ax.get_xlim()[1], curr_ax.get_ylim()[1])
        curr_ax.set_xlim(min_range, max_range)
        curr_ax.set_ylim(min_range, max_range)

        # plot linear regression
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

        # plot text
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

        # output
        if isinstance(save, str):
            name = "".join(i for i in self.name if i.isalnum())
            save_image(save, f"validation_{name}_kernel{kernel}.png")
        else:
            plt.show()

    def _fit(self, log: bool = False) -> None:
        optimizer = Optimizer(
            self,
            [self.table],
            epochs=50,
            patience=3,
            checkpoint=self.checkpoint,
            device=self.device.type,
            output=self.output,
        )
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
                optimizer.model.load_state_dict(curr_state_dict)
                loss = optimizer.train_simultaneous()
                if loss < best_loss:
                    best_loss = loss
                    best_state_dict = copy.deepcopy(
                        optimizer.model.state_dict()
                    )

        self.load_state_dict(best_state_dict)
        optimizer.save(self.checkpoint)

        if self.train_omega or self.train_theta:
            for mod in self.round.modules():
                if isinstance(mod, BindingMode):
                    for layer in mod.layers:
                        if isinstance(layer, Conv1d):
                            if self.train_omega:
                                layer.omega.requires_grad_()
                            if self.train_theta:
                                layer.theta.requires_grad_()
                elif isinstance(mod, BindingCooperativity):
                    if self.train_omega:
                        mod.omega.requires_grad_()

            optimizer.train_simultaneous()
            optimizer.save(self.checkpoint)


class Fit(_Fit):
    """Curve fitting to energy data [observation(y) ~ prediction(logZ)]"""

    def __init__(
        self,
        rnd: _ARound | Aggregate,
        dataset: CountTable,
        prediction: Callable[[Tensor], Tensor],
        observation: Callable[[Tensor], Tensor] = lambda x: x,
        update_construct: bool = False,
        train_offset: bool = False,
        train_omega: bool = False,
        train_theta: bool = False,
        train_hill: bool = False,
        max_split: int | None = None,
        checkpoint: str | os.PathLike[str] = "valmodel.pt",
        output: str | os.PathLike[str] = os.devnull,
        device: str | None = None,
        name: str = "",
    ) -> None:
        super().__init__(
            rnd,
            dataset,
            prediction=prediction,
            observation=observation,
            update_construct=update_construct,
            train_omega=train_omega,
            train_theta=train_theta,
            train_hill=train_hill,
            max_split=max_split,
            checkpoint=checkpoint,
            output=output,
            device=device,
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
        save: bool = False,
    ) -> None:
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
            save=save,
        )

    def fit(self) -> None:
        super()._fit(log=False)


class LogFit(_Fit):
    """Curve fitting to binding data [log observation(y) ~ prediction(logZ)]"""

    def __init__(
        self,
        rnd: _ARound | Aggregate,
        dataset: CountTable,
        prediction: Callable[[Tensor], Tensor],
        observation: Callable[[Tensor], Tensor] = lambda x: x,
        update_construct: bool = False,
        train_offset: bool = False,
        train_omega: bool = False,
        train_theta: bool = False,
        train_hill: bool = False,
        max_split: int | None = None,
        checkpoint: str | os.PathLike[str] = "valmodel.pt",
        output: str | os.PathLike[str] = os.devnull,
        device: str | None = None,
        name: str = "",
    ) -> None:
        super().__init__(
            rnd,
            dataset,
            prediction=prediction,
            observation=observation,
            update_construct=update_construct,
            train_omega=train_omega,
            train_theta=train_theta,
            train_hill=train_hill,
            max_split=max_split,
            checkpoint=checkpoint,
            output=output,
            device=device,
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
        save: bool = False,
    ) -> None:
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
            save=save,
        )

    def fit(self) -> None:
        super()._fit(log=True)
