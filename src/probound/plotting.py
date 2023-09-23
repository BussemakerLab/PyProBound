"""Plotters for sequence logos and diagnostics"""
import copy
import math
import os
import warnings
from collections.abc import Collection
from typing import Any, cast

import logomaker
import matplotlib
import matplotlib.collections
import matplotlib.figure
import matplotlib.font_manager
import numpy as np
import pandas as pd
import scipy
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from torch import Tensor

from .aggregate import Aggregate, Contribution
from .binding import BindingMode
from .conv1d import Conv0d, Conv1d
from .cooperativity import BindingCooperativity, Spacing
from .experiment import Experiment
from .psam import PSAM
from .rounds import ERound, _ARound
from .table import Batch, score
from .utils import avg_pool1d

matplotlib.rcParams["font.sans-serif"] = "Arial"
gnbu = plt.get_cmap("YlGnBu")(range(256))[64:]
gnbu_mod = matplotlib.colors.LinearSegmentedColormap.from_list(
    "gnbu_mod", gnbu
)
cmap = matplotlib.colormaps["bwr"].copy()
cmap.set_bad(color="gray")


def count_kmers(sequences: Collection[str], kmer_length: int = 3) -> Tensor:
    """Returns sparse count matrix of k-mers in list of sequences"""
    vocabulary: dict[str, int] = {}
    data: list[int] = []
    indices: list[int] = []
    indptr: list[int] = [0]
    for seq in sequences:
        count: dict[int, int] = {}
        for view in range(0, len(seq) - kmer_length + 1):
            kmer = seq[view : view + kmer_length]

            # get key from vocabulary
            if kmer not in vocabulary:
                key = len(vocabulary)
                vocabulary[kmer] = key
            else:
                key = vocabulary[kmer]

            # running sum of kmers in sequence
            if key not in count:
                count[key] = 1
            else:
                count[key] += 1

        # add to csc initializers
        indptr.append(indptr[-1] + len(count))
        data.extend(count.values())
        indices.extend(count.keys())

    return torch.sparse_csc_tensor(
        indptr,
        indices,
        data,
        size=(len(vocabulary), len(sequences)),
        dtype=torch.float32,
    )


def save_image(save: str, filename: str) -> None:
    filename = os.path.join(save, filename)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def logo(
    psam: PSAM,
    rev: bool = False,
    fix_gauge: bool = True,
    save: bool | str = False,
) -> None:
    """Plots a sequence logo for the given PSAM"""
    if psam.out_channels // psam.n_strands != 1:
        raise ValueError("Cannot plot logo for multi-channel PSAMs")
    if psam.alphabet is None:
        raise ValueError("Cannot plot logo for PSAMs without alphabets")

    if fix_gauge:
        psam = copy.deepcopy(psam)
        psam.fix_gauge()

    matrices = [
        cast(
            NDArray[Any],
            torch.movedim(psam.get_filter(i).detach(), -1, 1)[0].cpu().numpy(),
        )
        for i in range(psam.interaction_distance + 1)
    ]

    # binding mode attributes
    in_channels = psam.in_channels
    size = len(psam.symmetry)
    positions = in_channels * np.arange(size) + (in_channels / 2) - 0.5

    # set up subplots
    interaction = psam.interaction_distance > 0
    width = 7 if interaction else 8
    logo_height = width / 4
    colorbar_width = width / 20
    if interaction:
        fig, axs = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(width + colorbar_width, width + logo_height),
            gridspec_kw={
                "height_ratios": [logo_height, width],
                "width_ratios": [width, colorbar_width],
            },
        )
        axs[0, 1].axis("off")
        axs[1, 1].axis("off")
    else:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(width, logo_height))

    # set up mononucleotide matrix
    matrix = matrices[0]
    matrix -= matrix.mean(1, keepdims=True)
    matrix = np.flip(matrix, axis=(0, 1)) if rev else matrix

    if psam.alphabet is not None:
        # create sequence logo
        dataframe = pd.DataFrame(matrix, columns=psam.alphabet.alphabet)
        dataframe.columns = dataframe.columns.astype(str)
        if "Helvetica" in matplotlib.font_manager.findfont("Helvetica"):
            font_name = "Helvetica"
        else:
            font_name = "DejaVu Sans"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            energy_logo = logomaker.Logo(
                dataframe,
                ax=axs[0, 0] if interaction else axs,
                shade_below=0.5,
                fade_below=0.5,
                font_name=font_name,
                color_scheme=psam.alphabet.color_scheme,
            )
        energy_logo.style_spines(visible=False)
        energy_logo.style_spines(spines=["left", "bottom"], visible=True)
        energy_logo.ax.set_ylabel(r"$-\Delta \Delta$G/RT", labelpad=-1)
        energy_logo.ax.set_xlabel("position", labelpad=-1)
        energy_logo.ax.xaxis.set_ticks_position("none")
        energy_logo.ax.xaxis.set_tick_params(pad=-1)
    else:
        max_val = max(np.nanmax(matrix), np.nanmax(-matrix), 1e-8)
        axs_curr = axs[0, 0] if interaction else axs
        axs_curr.imshow(
            matrix.T,
            interpolation="none",
            cmap=cmap,
            vmin=-max_val,
            vmax=max_val,
            aspect="auto",
        )
        axs_curr.set_ylabel("channels")
        axs_curr.set_xlabel("position")

    # create interaction logo
    if interaction:
        # create empty heatmap matrix
        heatmap = np.empty((size * in_channels, size * in_channels))
        heatmap[:] = np.NaN

        # fill heatmap matrix
        for dist in range(1, len(matrices)):
            matrix = matrices[dist]
            for pos in range(size - dist):
                x, y = (pos + dist) * in_channels, pos * in_channels
                heatmap[x : x + in_channels, y : y + in_channels] = matrix[pos]
                heatmap[y : y + in_channels, x : x + in_channels] = matrix[
                    pos
                ].T

        if rev:
            heatmap = np.flipud(np.fliplr(heatmap))

        # draw heatmap
        max_val = max(
            1e-15,
            cast(float, np.nanmax(heatmap)),
            cast(float, np.nanmax(-heatmap)),
        )
        axs[1, 0].imshow(
            heatmap,
            interpolation="none",
            cmap=cmap,
            vmin=-max_val,
            vmax=max_val,
        )

        # draw labels
        positions = in_channels * np.arange(size) + (in_channels / 2) - 0.5
        labels = np.arange(size)
        axs[1, 0].set_xticks(positions, labels)
        axs[1, 0].set_yticks(positions, labels)

        # draw colorbar
        colorbar = plt.cm.ScalarMappable(cmap=cmap)
        colorbar.set_clim(-max_val, max_val)
        fig.colorbar(colorbar, ax=axs[1, 1], fraction=1, pad=0)

    # add title
    title = psam.name
    if rev:
        title += ", Reversed"
    if interaction:
        fig.suptitle(title, y=1)
    else:
        plt.title(title)

    # output
    plt.tight_layout()
    if isinstance(save, str):
        name = "".join(i for i in psam.name if i.isalnum())
        save_image(save, f"logo_{name}{'_rev' if rev else ''}.png")
    else:
        plt.show()


def cooperativity(
    spacing_matrix: Spacing | BindingCooperativity, save: bool | str = False
) -> None:
    fig = plt.figure(figsize=(6.4, 4.8), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, width_ratios=[1, 0.1])
    axs = subfigs[0].subplots(
        spacing_matrix.n_strands,
        spacing_matrix.n_strands,
        sharex=True,
        sharey=True,
        gridspec_kw={"right": 0.15},
    )

    # get spacing matrix and axis labels
    if isinstance(spacing_matrix, Spacing):
        out = torch.exp(
            spacing_matrix.get_spacing(
                spacing_matrix.max_num_windows, spacing_matrix.max_num_windows
            )
            .detach()
            .cpu()
        )
        cast(matplotlib.figure.SubFigure, subfigs[0]).supxlabel(
            "-".join([i.name for i in spacing_matrix.binding_mode_key_0])
        )
        cast(matplotlib.figure.SubFigure, subfigs[0]).supylabel(
            "-".join([i.name for i in spacing_matrix.binding_mode_key_1])
        )
    else:
        out = torch.exp(spacing_matrix.get_spacing().detach().cpu()).squeeze()
        cast(matplotlib.figure.SubFigure, subfigs[0]).supxlabel(
            spacing_matrix.binding_mode_0.name
        )
        cast(matplotlib.figure.SubFigure, subfigs[0]).supylabel(
            spacing_matrix.binding_mode_1.name
        )

    # draw subplots
    max_val = max(np.nanmax(abs(1 - out)), 1e-7)
    norm = matplotlib.colors.LogNorm(
        vmin=np.exp(-max_val), vmax=np.exp(max_val)
    )
    for n_strand_0, strand_0 in enumerate(
        torch.chunk(out, spacing_matrix.n_strands, dim=0)
    ):
        for n_strand_1, strand_1 in enumerate(
            torch.chunk(strand_0, spacing_matrix.n_strands, dim=1)
        ):
            ax = cast(plt.Axes, axs[n_strand_0, n_strand_1])
            ax.imshow(strand_1, interpolation="none", cmap=cmap, norm=norm)
            ax.xaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(integer=True)
            )
            ax.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(integer=True)
            )

    # label title and axes
    cast(matplotlib.figure.SubFigure, subfigs[0]).suptitle(
        f"{spacing_matrix.name} Spacing"
    )
    if spacing_matrix.n_strands == 2:
        cast(plt.Axes, axs[0, 0]).set_ylabel("Forward")
        cast(plt.Axes, axs[1, 0]).set_ylabel("Reverse")
        cast(plt.Axes, axs[1, 0]).set_xlabel("Forward")
        cast(plt.Axes, axs[1, 1]).set_xlabel("Reverse")

    # draw colorbar
    bar_axs = subfigs[1].add_axes(
        [0, 0.143, 1, 0.795]  # left  # bottom  # width  # height
    )
    bar_axs.axis("off")
    colorbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cast(matplotlib.figure.SubFigure, subfigs[1]).colorbar(
        colorbar, ax=bar_axs, fraction=1, pad=0
    )

    # output
    if isinstance(save, str):
        save_image(save, f"{spacing_matrix}_spacing.png")
    else:
        plt.show()


def omega(
    conv1d: Conv0d | Conv1d | BindingMode, save: bool | str = False
) -> None:
    if isinstance(conv1d, BindingMode):
        if not isinstance(conv1d.layers[0], (Conv1d, Conv0d)):
            raise ValueError("Cannot make plot if first layer is not Conv1d")
        conv1d = conv1d.layers[0]

    for i_out, out in enumerate(conv1d.get_omega().detach().cpu().unbind(1)):
        out = torch.exp(out[conv1d.min_input_length :])
        rows = int(10 * (out.shape[0] / sum(out.shape)))
        rows = min(8, max(2, rows))
        columns = 10 - rows
        if rows < columns:
            fig, axs = plt.subplots(
                nrows=2,
                figsize=(columns, rows),
                constrained_layout=True,
                gridspec_kw={"height_ratios": [1000, 1]},
            )
        else:
            fig, axs = plt.subplots(
                ncols=2,
                figsize=(rows, columns),
                constrained_layout=True,
                gridspec_kw={"width_ratios": [1000, 1]},
            )
        axs[1].axis("off")
        max_val = max(np.nanmax(out.log().abs()), 1e-7)
        norm = matplotlib.colors.LogNorm(
            vmin=np.exp(-max_val), vmax=np.exp(max_val)
        )
        axs[0].imshow(
            out, interpolation="none", cmap=cmap, norm=norm, aspect="auto"
        )
        axs[0].set_xlabel("Position on probe")
        axs[0].set_ylabel("Length")
        axs[0].set_title(f"{conv1d.layer_spec.name} Omega (O={i_out})")

        # add labels
        axs[0].xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True)
        )
        axs[0].yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True)
        )
        positions = axs[0].get_xticks()
        step = int(positions[-1] - positions[-2])
        if step > 0:
            positions = range(
                0, conv1d.max_input_length - conv1d.min_input_length + 1, step
            )
            labels = [conv1d.min_input_length + i for i in positions]
            axs[0].set_yticks(positions, labels)

        # draw colorbar
        colorbar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(
            colorbar,
            ax=axs,
            fraction=1,
            pad=0,
            location="bottom" if rows < columns else "right",
        )

        # output
        if isinstance(save, str):
            save_image(save, f"{conv1d}_O{i_out}_omega.png")
        else:
            plt.show()


def theta(conv1d: Conv1d | BindingMode, save: bool | str = False) -> None:
    if isinstance(conv1d, BindingMode):
        if not isinstance(conv1d.layers[0], Conv1d):
            raise ValueError("Cannot make plot if first layer is not Conv1d")
        conv1d = conv1d.layers[0]
    for i_length, length in enumerate(
        conv1d.get_theta().detach().cpu()[conv1d.min_input_length :],
        start=conv1d.min_input_length,
    ):
        for i_out, out in enumerate(length):
            fig, axs = plt.subplots(figsize=(5, 5), constrained_layout=True)
            plt.imshow(out.T, interpolation="none", cmap=cmap, aspect="auto")
            plt.xlabel("Position on probe")
            plt.ylabel("Position on PSAM")
            plt.title(
                f"{conv1d.layer_spec.name} Theta (L={i_length}, O={i_out})"
            )
            axs.xaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(integer=True)
            )
            axs.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(integer=True)
            )

            # draw colorbar
            colorbar = plt.cm.ScalarMappable(cmap=cmap)
            fig.colorbar(colorbar, location="bottom")

            # output
            if isinstance(save, str):
                save_image(save, f"{conv1d}_L{i_length}O{i_out}_theta.png")
            else:
                plt.show()


def _enrichment(
    counts_obs: Tensor,
    counts_pred: Tensor,
    columns: list[int],
    kernel: int = 1,
    title: str = "",
) -> matplotlib.figure.Figure:
    """Make enrichment scatterplot"""

    if counts_obs.shape != counts_pred.shape:
        raise ValueError(
            f"counts_obs shape {counts_obs.shape} does not match"
            f" counts_pred shape {counts_pred.shape}"
        )

    if not 0 <= min(columns) <= max(columns) < counts_obs.shape[1]:
        raise ValueError(
            f"columns {columns} incompatible with"
            f" counts_obs shape {counts_obs.shape}"
        )

    # setup subplots
    hexbin = len(columns) <= 2 and len(counts_pred) / kernel > 500
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
        plt.yscale("log")
        plt.xscale("log")

    # plot binned enrichments
    min_range = float("inf")
    max_range = float("-inf")
    n_bins = 0
    for i in range(len(columns) - 1):
        # get the columns
        col1 = columns[i]
        col2 = columns[i + 1]
        cols_pred = counts_pred[:, [col1, col2]]
        cols_obs = counts_obs[:, [col1, col2]]

        # remove rows which have 0's for both columns
        nonzero = torch.any(cols_obs > 0, dim=1)
        cols_pred, cols_obs = cols_pred[nonzero], cols_obs[nonzero]

        # sort rows by predicted fold enrichment
        fold_enr = cols_pred[:, 1] / cols_pred[:, 0]
        sorting = torch.argsort(fold_enr, descending=True)
        cols_pred, cols_obs = cols_pred[sorting], cols_obs[sorting]

        # bin the rows with AvgPool1D
        if kernel > 1:
            eps = 1 / kernel
            cols_pred = avg_pool1d(cols_pred, kernel)
            cols_obs = avg_pool1d(cols_obs, kernel)
        else:
            eps = 1
        n_bins += len(cols_obs)

        # calculate enrichment
        x_pred = (cols_pred[:, 1] + eps) / (cols_pred[:, 0] + eps)
        y_obs = (cols_obs[:, 1] + eps) / (cols_obs[:, 0] + eps)

        # update range
        min_range = min(min_range, x_pred.min().item(), y_obs.min().item())
        max_range = max(max_range, x_pred.max().item(), y_obs.max().item())

        # plot and print statistics
        spearman = scipy.stats.spearmanr(x_pred, y_obs).statistic
        pearson = scipy.stats.pearsonr(x_pred.log(), y_obs.log()).statistic
        rmsle = (x_pred.log() - y_obs.log()).square().mean().sqrt().item()
        label = f"{col1}→{col2}"
        label += (
            rf" $R\in$ [{y_obs.min().item():.2e},{y_obs.max().item():.2e}]"
            f"\n$r_s$={spearman:.3f}, $r$={pearson:.3f}, RMSLE={rmsle:.3f}"
        )
        if hexbin:
            binplot = axs[0].hexbin(
                x_pred,
                y_obs,
                cmap=gnbu_mod,
                label=label,
                bins="log",
                xscale="log",
                yscale="log",
            )
            fig.colorbar(binplot, ax=axs[1], fraction=1, pad=0)
        else:
            axs.scatter(x_pred, y_obs, label=label, alpha=0.5)

    # make square
    curr_ax = axs[0] if hexbin else axs
    curr_ax.plot([min_range, max_range], [min_range, max_range], "k--")
    curr_ax.set_xlim(0.9 * min_range, 1.1 * max_range)
    curr_ax.set_ylim(0.9 * min_range, 1.1 * max_range)

    # add title, legend, labels, and colorbar
    if kernel > 1:
        title += f" ({n_bins:,} bins of n={kernel})"
    else:
        title += f" ({n_bins:,} probes)"
    curr_ax.set_title(title)
    curr_ax.set_xlabel("Predicted Enrichment")
    curr_ax.set_ylabel("Observed Enrichment")
    curr_ax.legend(loc="lower right")

    return fig


def probe_enrichment(
    experiment: Experiment,
    batch: Batch,
    columns: list[int] | None = None,
    kernel: int = 500,
    save: bool | str = False,
) -> None:
    """Plot enrichment of sequences, binned by predicted enrichment"""

    counts_obs, counts_pred = score(
        experiment, batch, fun="log_prediction", target=batch.target
    )
    counts_pred = counts_pred.exp()
    if columns is None:
        columns = list(range(counts_obs.shape[1]))

    # make plot
    _enrichment(
        counts_obs,
        counts_pred,
        columns=columns,
        kernel=kernel,
        title=f"{experiment.name} Probe-Level Enr.",
    )

    # output
    if isinstance(save, str):
        name = "".join(i for i in experiment.name if i.isalnum())
        cols = "-".join(map(str, columns))
        save_image(save, f"probeenr_{name}_cols{cols}_kernel{kernel}.png")
    else:
        plt.show()


def kmer_enrichment(
    experiment: Experiment,
    batch: Batch,
    columns: list[int] | None = None,
    kmer_length: int = 3,
    kernel: int = 500,
    save: bool | str = False,
) -> None:
    """Plot enrichment of sequences, binned by predicted enrichment"""

    counts_obs, counts_pred = score(
        experiment, batch, fun="log_prediction", target=batch.target
    )
    counts_pred = counts_pred.exp()
    kmer_counts = count_kmers(batch.seqs, kmer_length=kmer_length)
    counts_obs = kmer_counts @ counts_obs
    counts_pred = kmer_counts @ counts_pred
    if columns is None:
        columns = list(range(counts_obs.shape[1]))

    # make plot
    _enrichment(
        counts_obs,
        counts_pred,
        columns=columns,
        kernel=kernel,
        title=f"{experiment.name} {kmer_length}-mer Enr.",
    )

    # output
    if isinstance(save, str):
        name = "".join(i for i in experiment.name if i.isalnum())
        cols = "-".join(map(str, columns))
        save_image(
            save, f"{kmer_length}merenr_{name}_cols{cols}_kernel{kernel}.png"
        )
    else:
        plt.show()


def sequence_counts(
    experiment: Experiment,
    batch: Batch,
    column_index: int,
    sigmas: int = 1,
    save: bool | str = False,
) -> None:
    """Plot predicted vs. observed counts of sequences"""

    # get predicted and observed counts
    y_obs, x_pred = score(
        experiment, batch, fun="log_prediction", target=batch.target
    )
    x_pred = x_pred.exp()
    y_obs = y_obs[:, column_index]
    x_pred = x_pred[:, column_index]
    nonzero = y_obs > 0
    x_pred = x_pred[nonzero]
    y_obs = y_obs[nonzero]

    def a_sol(x: Tensor) -> Tensor:
        """A solution to x = y +/- stdev*sqrt(y)"""
        return 0.5 * (
            2 * x + sigmas**2 + math.sqrt(sigmas**4 + 4 * x * sigmas**2)
        )

    def b_sol(x: Tensor) -> Tensor:
        """B solution to x = y +/- stdev*sqrt(y)"""
        return 0.5 * (
            2 * x + sigmas**2 - math.sqrt(sigmas**4 + 4 * x * sigmas**2)
        )

    def lower_bound(x: Tensor) -> Tensor:
        return x - sigmas * torch.sqrt(x)

    def upper_bound(x: Tensor) -> Tensor:
        return x + sigmas * torch.sqrt(x)

    fig, axs = plt.subplots(
        ncols=2, figsize=(5.5, 5), gridspec_kw={"width_ratios": [5, 0.5]}
    )
    axs[1].axis("off")

    # plot hexbin and print statistics
    spearman = scipy.stats.spearmanr(x_pred, y_obs).statistic
    pearson = scipy.stats.pearsonr(x_pred.log(), y_obs.log()).statistic
    rmsle = (x_pred.log() - y_obs.log()).square().mean().sqrt().item()
    label = (
        rf" $R\in$ [{y_obs.min().item():.2e},{y_obs.max().item():.2e}]"
        f"\n$r_s$={spearman:.3f}, $r$={pearson:.3f}, RMSLE={rmsle:.3f}"
    )
    binplot = axs[0].hexbin(
        x_pred,
        y_obs,
        cmap=gnbu_mod,
        label=label,
        bins="log",
        xscale="log",
        yscale="log",
    )
    fig.colorbar(binplot, ax=axs[1], fraction=1, pad=0)

    # add expectation line
    min_range = torch.cat((x_pred, y_obs)).min()
    max_range = torch.cat((x_pred, y_obs)).max()
    min_observed = y_obs.min()
    axs[0].plot([min_observed, max_range], [min_observed, max_range], "k--")

    # plot standard deviation
    pos_ranges = torch.logspace(
        torch.log10(b_sol(min_observed)).item(),
        torch.log10(b_sol(max_range)).item(),
        steps=50,
    )
    neg_ranges = torch.logspace(
        torch.log10(a_sol(min_observed)).item(),
        torch.log10(a_sol(max_range)).item(),
        steps=50,
    )
    y_obs = torch.maximum(x_pred, torch.ones(1))
    frac_inside = sum(
        (y_obs > lower_bound(x_pred)) & (y_obs < upper_bound(x_pred))
    ) / len(y_obs)
    lines = matplotlib.collections.LineCollection(
        (
            list(zip(pos_ranges, upper_bound(pos_ranges))),
            list(zip(neg_ranges, lower_bound(neg_ranges))),
        ),
        label=rf"{sigmas}$\sigma$ ({frac_inside:.1%} within)",
        linestyle="dashed",
        color="red",
    )
    axs[0].add_collection(lines)

    # make square
    axs[0].set_ylim(0.9 * min_observed, 1.1 * max_range)
    axs[0].set_xlim(
        0.9
        * torch.cat(
            (min_range.unsqueeze(0), pos_ranges, upper_bound(pos_ranges))
        ).min(),
        1.1
        * torch.cat(
            (max_range.unsqueeze(0), neg_ranges, lower_bound(neg_ranges))
        ).max(),
    )

    # add title, legend, labels
    title = (
        f"{experiment.name} Col {column_index} Probe-Level Counts"
        f" ({len(x_pred):,} probes)"
    )
    axs[0].set_title(title)
    axs[0].set_xlabel("Predicted Count")
    axs[0].set_ylabel("Observed Count")
    axs[0].legend(loc="upper left")

    # output
    if isinstance(save, str):
        name = "".join(i for i in experiment.name if i.isalnum())
        save_image(save, f"counts_{name}_col{column_index}.png")
    else:
        plt.show()


def kd_consistency(
    experiment: Experiment,
    i_index: int,
    b_index: int,
    f_index: int,
    batch: Batch,
    kernel: int = 500,
    save: bool | str = False,
) -> None:
    # get Kd's
    i_round = experiment.rounds[i_index]
    b_round = experiment.rounds[b_index]
    f_round = experiment.rounds[f_index]
    free_protein = experiment.free_protein(i_index, b_index, f_index)
    k_a = torch.exp(
        score(b_round, batch, "log_aggregate")[1] - math.log(free_protein)
    )

    # get counts of all rounds
    counts_obs, counts_pred = score(
        experiment, batch, fun="log_prediction", target=batch.target
    )
    counts_pred = counts_pred.exp()

    # sort by Ka
    sorting = torch.argsort(k_a)
    k_a = k_a[sorting]
    counts_obs = counts_obs[sorting]
    counts_pred = counts_pred[sorting]

    # bin
    if kernel > 1:
        k_a = avg_pool1d(k_a, kernel)
        counts_obs = avg_pool1d(counts_obs, kernel)
        counts_pred = avg_pool1d(counts_pred, kernel)
        eps = 1 / kernel
    else:
        eps = 1.0

    # calculate bound fractions
    pred_bound = (
        (counts_pred[:, b_index] + eps) / (counts_pred[:, i_index] + eps)
    ) * torch.exp(i_round.log_eta - b_round.log_eta).detach().cpu()
    obs_bound = (
        (counts_obs[:, b_index] + eps) / (counts_obs[:, i_index] + eps)
    ) * torch.exp(i_round.log_eta - b_round.log_eta).detach().cpu()

    # calculate free fractions
    pred_free = (
        (counts_pred[:, f_index] + eps) / (counts_pred[:, i_index] + eps)
    ) * torch.exp(i_round.log_eta - f_round.log_eta).detach().cpu()
    obs_free = (
        (counts_obs[:, f_index] + eps) / (counts_obs[:, i_index] + eps)
    ) * torch.exp(i_round.log_eta - f_round.log_eta).detach().cpu()

    # plot
    fig, axs = plt.subplots(
        2, 1, figsize=(5, 5), constrained_layout=True, sharex=True
    )
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"Predicted $1/K_D$")

    rmse_bound = (pred_bound - obs_bound).square().mean().sqrt().item()
    axs[0].scatter(k_a, obs_bound, alpha=0.5, label=f"RMSE={rmse_bound:.3f}")
    axs[0].plot(k_a, pred_bound, "k--")
    axs[0].legend(loc="lower right")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Bound fraction")

    rmse_free = (pred_free - obs_free).square().mean().sqrt().item()
    axs[1].scatter(k_a, obs_free, alpha=0.5, label=f"RMSE={rmse_free:.3f}")
    axs[1].plot(k_a, pred_free, "k--")
    axs[1].legend(loc="upper right")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("Free fraction")

    title = r"$K_D$ Model Consistency"
    if kernel > 1:
        title += f" ({len(counts_obs):,} bins of n={kernel})"
    else:
        title += f" ({len(counts_obs):,} probes)"
    fig.suptitle(title)

    # output
    if isinstance(save, str):
        name = "".join(i for i in experiment.name if i.isalnum())
        columns = "-".join(str(i) for i in (i_index, b_index, f_index))
        save_image(save, f"kd_consistency_{name}_col{columns}.png")
    else:
        plt.show()


def keff_consistency(
    experiment: Experiment,
    batch: Batch,
    columns: list[int] | None = None,
    kernel: int = 500,
    save: bool | str = False,
) -> None:
    if columns is None:
        columns = [
            i
            for i, rnd in enumerate(experiment.rounds)
            if isinstance(rnd, ERound)
        ]

    _, axs = plt.subplots(figsize=(6, 3), constrained_layout=True)

    # get counts of all rounds
    counts_obs, counts_pred = score(
        experiment, batch, fun="log_prediction", target=batch.target
    )
    counts_pred = counts_pred.exp()

    n_bins = 0
    for e_col in columns:
        # get rounds
        e_round = experiment.rounds[e_col]
        i_col = next(
            i
            for i, rnd in enumerate(experiment.rounds)
            if rnd is e_round.reference_round
        )
        i_round = experiment.rounds[i_col]

        # get k_eff
        k_eff = torch.exp(score(e_round, batch, "log_aggregate")[1])

        # get counts
        cols_pred = counts_pred[:, [i_col, e_col]]
        cols_obs = counts_obs[:, [i_col, e_col]]

        # remove rows with no counts
        nonzero = torch.any(cols_obs > 0, dim=1)
        cols_pred = cols_pred[nonzero]
        cols_obs = cols_obs[nonzero]
        k_eff = k_eff[nonzero]

        # sort by k_eff
        sorting = torch.argsort(k_eff)
        k_eff = k_eff[sorting]
        cols_obs = cols_obs[sorting]
        cols_pred = cols_pred[sorting]

        # bin
        if kernel > 1:
            k_eff = avg_pool1d(k_eff, kernel)
            cols_obs = avg_pool1d(cols_obs, kernel)
            cols_pred = avg_pool1d(cols_pred, kernel)
            eps = 1 / kernel
        else:
            eps = 1.0
        n_bins += len(cols_obs)

        # calculate bound fractions
        pred_modified = (
            (cols_pred[:, 1] + eps) / (cols_pred[:, 0] + eps)
        ) * torch.exp(i_round.log_eta - e_round.log_eta).detach().cpu()
        obs_modified = (
            (cols_obs[:, 1] + eps) / (cols_obs[:, 0] + eps)
        ) * torch.exp(i_round.log_eta - e_round.log_eta).detach().cpu()

        # plot
        rmse = (pred_modified - obs_modified).square().mean().sqrt().item()
        label = f"{e_round} RMSE={rmse:.3f}"
        axs.scatter(k_eff, obs_modified, alpha=0.5, label=label)
        axs.plot(k_eff, pred_modified, "k--")
        axs.set_xscale("log")
        axs.set_yscale("log")
        axs.set_xlabel(r"Predicted $k_{\mathrm{eff}}$")
        axs.set_ylabel("Bound fraction")
        axs.set_title(r"$k_{\mathrm{eff}}$ Model Consistency")
        axs.legend(loc="lower right")

    title = r"$k_{\mathrm{eff}}$ Model Consistency"
    if kernel > 1:
        title += f" ({n_bins:,} bins of n={kernel})"
    else:
        title += f" ({n_bins:,} probes)"
    axs.set_title(title)

    # output
    if isinstance(save, str):
        name = "".join(i for i in experiment.name if i.isalnum())
        cols = "-".join(map(str, columns))
        save_image(save, f"kd_consistency_{name}_cols{cols}.png")
    else:
        plt.show()


def contribution(
    rnd: _ARound | Aggregate,
    batch: Batch,
    kernel: int = 500,
    save: bool | str = False,
) -> None:
    """Plot binding mode contributions as a function of selection strength"""

    _, log_aggregate = score(
        rnd,
        batch,
        "forward" if isinstance(rnd, Aggregate) else "log_aggregate",
    )

    bmd_names: list[str] = []
    bmd_contributions_list: list[Tensor] = []
    for ctrb in [m for m in rnd.modules() if isinstance(m, Contribution)]:
        bmd = ctrb.binding
        bmd_names.append("-".join(i.name for i in bmd.key()))
        bmd_contributions_list.append(score(ctrb, batch)[1].unsqueeze(1))
    bmd_contributions = torch.cat(bmd_contributions_list, dim=1)

    fig, axs = plt.subplots(
        2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": (1, 4)}
    )

    # sort
    sorting = torch.argsort(log_aggregate, dim=0, descending=False)
    log_aggregate = log_aggregate[sorting]
    bmd_contributions = bmd_contributions[sorting]
    bmd_contributions -= bmd_contributions.logsumexp(dim=1, keepdim=True)

    # bin
    bin_contributions = avg_pool1d(torch.exp(bmd_contributions), kernel)
    bin_partition = avg_pool1d(torch.exp(log_aggregate), kernel)
    x_min, x_max = torch.min(bin_partition), torch.max(bin_partition)

    # stack plot
    labels = [
        f"{name} ({contribution:.0%})"
        for name, contribution in zip(
            bmd_names, torch.mean(bin_contributions, dim=0)
        )
    ]
    axs[1].stackplot(bin_partition, bin_contributions.T, labels=labels)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Aggregate")
    axs[1].set_ylabel("Contribution")
    axs[1].legend(loc="lower right")
    axs[1].set_ylim((0, 1))
    axs[1].set_xlim((x_min, x_max))

    # density plot
    xrange = np.logspace(
        torch.log10(x_min),
        torch.log10(x_max),
        math.ceil(len(log_aggregate) / kernel) if kernel > 1 else 1_000,
    )
    axs[0].hist(torch.exp(log_aggregate), xrange, log=True)
    axs[0].set_xscale("log")
    axs[0].set_ylabel("Count")
    axs[0].set_xlim((x_min, x_max))
    axs[0].get_xaxis().set_visible(False)

    # add title
    title = f"{rnd} Mode Contributions"
    if kernel > 1:
        title += f" ({len(bin_partition):,} bins of n={kernel})"
    else:
        title += f" ({len(bin_partition):,} probes)"
    axs[0].set_title(title)
    fig.align_ylabels(axs)

    # output
    if isinstance(save, str):
        name = "".join(i for i in rnd.name if i.isalnum())
        save_image(save, f"contribution_{name}_kernel{kernel}.png")
    else:
        plt.show()
