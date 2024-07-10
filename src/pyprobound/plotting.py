"""Module of miscellaneous plotting functions."""

import copy
import math
import warnings
from typing import Any, TypeAlias, cast

import logomaker
import matplotlib
import matplotlib.collections
import matplotlib.font_manager
import numpy as np
import pandas as pd
import scipy
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray
from torch import Tensor

from .aggregate import Aggregate, Contribution
from .cooperativity import Cooperativity, Spacing
from .experiment import Experiment
from .layers import PSAM, Conv0d, Conv1d
from .mode import Mode
from .rounds import BaseRound, ExponentialRound
from .table import CountBatch, score
from .utils import avg_pool1d, count_kmers

if "Arial" in matplotlib.font_manager.findfont("Arial"):
    matplotlib.rcParams["font.sans-serif"] = "Arial"
gnbu = plt.get_cmap("YlGnBu")(range(256))[64:]
gnbu_mod = matplotlib.colors.LinearSegmentedColormap.from_list(
    "gnbu_mod", gnbu
)
cmap = matplotlib.colormaps["bwr"].copy()
cmap.set_bad(color="gray")

AxesArray: TypeAlias = NDArray[Any]


def logomaker_plotter(
    ax: Axes, psam: PSAM, reverse: bool = False, **kwargs: Any
) -> logomaker.Logo:
    """Plots the monomer sequence logo for the given PSAM using Logomaker.

    Args:
        ax: The Axes to draw to.
        psam: A PSAM to plot into a logo.
        reverse: Whether to plot the reverse complement.
    """
    if psam.out_channels // psam.n_strands != 1:
        raise ValueError("Cannot plot logo for multi-channel PSAMs")
    if psam.alphabet is None:
        raise ValueError("Cannot plot logo for PSAMs without alphabets")

    # Create monomer dataframe
    matrix: NDArray[np.float32] = (
        psam.get_filter(0)
        .detach()[0]
        .T.to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    matrix -= matrix.mean(1, keepdims=True)
    matrix = np.flip(matrix, axis=(0, 1)) if reverse else matrix
    dataframe = pd.DataFrame(matrix, columns=psam.alphabet.alphabet)
    dataframe.columns = dataframe.columns.astype(str)

    # Set font
    if "Helvetica" in matplotlib.font_manager.findfont("Helvetica"):
        font_name = "Helvetica"
    else:
        font_name = "DejaVu Sans"

    # Draw PSAM with Logomaker
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = logomaker.Logo(
            dataframe,
            ax=ax,
            shade_below=0.5,
            fade_below=0.5,
            font_name=font_name,
            color_scheme=psam.alphabet.color_scheme,
            **kwargs,
        )

    # Adjust labels
    out.style_spines(visible=False)
    out.style_spines(spines=["left", "bottom"], visible=True)
    ax.set_ylabel(r"$-\Delta \Delta$G/RT", labelpad=-1)
    labels = np.arange(psam.kernel_size)
    ax.set_xticks(labels, labels)

    return out


def pairwise_plotter(
    ax: Axes, psam: PSAM, reverse: bool = False, **kwargs: Any
) -> matplotlib.image.AxesImage:
    """Plots the pairwise heatmap for the given PSAM.

    Args:
        ax: The Axes to draw to.
        psam: A PSAM to plot into a logo.
        reverse: Whether to plot the reverse complement.
    """
    matrices = [
        cast(
            NDArray[np.float32],
            torch.movedim(psam.get_filter(i).detach(), -1, 1)[0]
            .to(device="cpu", dtype=torch.float32)
            .numpy(),
        )
        for i in range(psam.pairwise_distance + 1)
    ]

    # Binding mode attributes
    in_channels = psam.in_channels
    size = len(psam.symmetry)

    # Create empty heatmap matrix
    heatmap = np.empty((size * in_channels, size * in_channels))
    heatmap[:] = np.NaN

    # Fill heatmap matrix
    for dist in range(1, len(matrices)):
        matrix = matrices[dist]
        for pos in range(size - dist):
            x, y = (pos + dist) * in_channels, pos * in_channels
            heatmap[x : x + in_channels, y : y + in_channels] = matrix[pos]
            heatmap[y : y + in_channels, x : x + in_channels] = matrix[pos].T

    if reverse:
        heatmap = np.flipud(np.fliplr(heatmap))

    # Draw labels
    positions = in_channels * np.arange(size) + (in_channels / 2) - 0.5
    labels = np.arange(size)
    ax.set_xticks(positions, labels)
    ax.set_yticks(positions, labels)

    # Draw heatmap
    max_val = max(
        1e-15,
        cast(float, np.nanmax(heatmap)),
        cast(float, np.nanmax(-heatmap)),
    )
    return ax.imshow(
        heatmap,
        interpolation="none",
        cmap=cmap,
        vmin=-max_val,
        vmax=max_val,
        **kwargs,
    )


def logo(
    psam: PSAM,
    logo_height: int = 2,
    width: int = 8,
    reverse: bool = False,
    fix_gauge: bool = True,
) -> None:
    """Plots a sequence recognition logo for the given PSAM.

    Args:
        psam: A PSAM to plot into a logo.
        reverse: Whether to plot the reverse complement.
        fix_gauge: Whether to call fix_gauge() before plotting the logo.
    """
    if psam.out_channels // psam.n_strands != 1:
        raise ValueError("Cannot plot logo for multi-channel PSAMs")
    if psam.alphabet is None:
        raise ValueError("Cannot plot logo for PSAMs without alphabets")

    if fix_gauge:
        psam = copy.deepcopy(psam)
        psam.fix_gauge()

    # Set up subplots
    pairwise = psam.pairwise_distance > 0
    colorbar_width = width / 20
    if pairwise:
        fig, ax = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(width + colorbar_width, width + logo_height),
            gridspec_kw={
                "height_ratios": [logo_height, width],
                "width_ratios": [width, colorbar_width],
            },
            tight_layout=True,
        )
        axs = cast(AxesArray, ax)
        axs[0, 1].axis("off")
        axs[1, 1].axis("off")
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, logo_height))
        ax = cast(Axes, ax)

    # Create monomer logo
    logomaker_plotter(axs[0, 0] if pairwise else ax, psam, reverse)

    # Create pairwise logo
    if pairwise:
        heatmap = pairwise_plotter(axs[1, 0], psam, reverse)
        fig.colorbar(heatmap, ax=axs[1, 1], fraction=1, pad=0)

    # Add title
    title = psam.name
    if reverse:
        title += ", Reversed"
    if pairwise:
        fig.suptitle(title, y=1)
    else:
        plt.title(title)


def posbias(conv1d: Conv0d | Conv1d | Mode) -> None:
    r"""Plots the position bias profile :math:`\omega(x)`.

    Args:
        conv1d: A component containing a position bias profile.
    """
    if isinstance(conv1d, Mode):
        indices = [
            i
            for i, layer in enumerate(conv1d.layers)
            if isinstance(layer, Conv1d)
        ]
        if len(indices) != 1:
            raise ValueError(
                f"Mode {conv1d} does not have exactly 1 Conv1d layers"
            )
        conv1d = cast(Conv1d, conv1d.layers[indices[0]])

    # Get position bias for each output channel
    out_list = list(
        conv1d.get_log_posbias()
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .unbind(1)
    )
    if isinstance(conv1d, Conv0d) or conv1d.length_specific_bias:
        out_list = [out[conv1d.min_input_length :] for out in out_list]

    # Create colornorm
    max_val = max(max(np.nanmax(out.abs()) for out in out_list), 1e-7)
    norm = matplotlib.colors.LogNorm(
        vmin=np.exp(-max_val), vmax=np.exp(max_val)
    )

    # Create figure
    rows = sum(len(out) for out in out_list)
    columns = len(out_list[0][0])
    figsize = (min(max(columns, 2), 10), 1.5 * min(max(rows, 1.5), 10))
    if figsize[1] > 2 * figsize[0]:
        figsize = (figsize[0], 2 * figsize[0])
    fig, ax = plt.subplots(
        nrows=len(out_list),
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )
    if len(out_list) == 1:
        ax = np.array([ax])
    axs = cast(AxesArray, ax)

    # Add each output channel to figure
    for i_out, out in enumerate(out_list):
        heatmap = axs[i_out].imshow(
            torch.exp(out),
            interpolation="none",
            cmap=cmap,
            norm=norm,
            aspect="auto",
        )
        if len(out_list) > 1:
            axs[i_out].set_title(f"Out channel {i_out}")

        # Add tick labels
        if columns > 1:
            axs[i_out].xaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(integer=True)
            )
        else:
            axs[i_out].set_xticks([])
        axs[i_out].yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True)
        )
        positions = axs[i_out].get_yticks()
        step = int(positions[-1] - positions[-2])
        if step > 0 and (
            isinstance(conv1d, Conv0d) or conv1d.length_specific_bias
        ):
            axs[i_out].set_ylabel("Probe length")
            positions = range(
                0, conv1d.max_input_length - conv1d.min_input_length + 1, step
            )
            labels = [conv1d.min_input_length + i for i in positions]
            axs[i_out].set_yticks(positions, labels)
        else:
            axs[i_out].set_yticks([])

    # Figure-level labels
    if columns > 1:
        axs[-1].set_xlabel("Position on probe")
    fig.suptitle(f"{conv1d.layer_spec.name} posbias")

    # Draw colorbar
    fig.colorbar(
        heatmap,
        ax=axs.ravel().tolist(),
        fraction=1,
        location="bottom" if rows < columns else "right",
    )


def cooperativity(
    spacing_matrix: Spacing | Cooperativity, len_a: int = -1, len_b: int = -1
) -> None:
    r"""Plots the cooperativity position bias :math:`\omega_{a:b}(x^a, x^b)`.

    Args:
        spacing_matrix: A component containing the cooperativity position bias.
        len_a: The input length for mode `a`, if length-specific position bias.
        len_b: The input length for mode `b`, if length-specific position bias.
    """
    fig, ax = plt.subplots(
        spacing_matrix.n_strands,
        spacing_matrix.n_strands,
        figsize=(6, 5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if spacing_matrix.n_strands == 1:
        ax = np.array([[ax]])
    axs = cast(AxesArray, ax)

    # Get spacing matrix and axis labels
    if isinstance(spacing_matrix, Spacing):
        out = torch.exp(
            spacing_matrix.get_log_spacing_matrix(
                spacing_matrix.max_num_windows, spacing_matrix.max_num_windows
            )
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )
        fig.supylabel("-".join([i.name for i in spacing_matrix.mode_key_a]))
        fig.supxlabel("-".join([i.name for i in spacing_matrix.mode_key_b]))
    else:
        out = torch.exp(spacing_matrix.get_log_spacing_matrix().detach().cpu())
        out = out[len_a, len_b, :, :, :, :]
        fig.supylabel(spacing_matrix.mode_a.name)
        fig.supxlabel(spacing_matrix.mode_b.name)

    # Draw subplots
    max_val = max(np.nanmax(out.log().nan_to_num(0, 0, 0).abs()), 1e-7)
    norm = matplotlib.colors.LogNorm(
        vmin=np.exp(-max_val), vmax=np.exp(max_val)
    )
    for ax_0, strand_0 in zip(ax, out):
        for ax, strand_1 in zip(ax_0, strand_0):
            heatmap = ax.imshow(
                strand_1,
                interpolation="none",
                cmap=cmap,
                norm=norm,
                aspect="equal",
            )
            ax.xaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(integer=True)
            )
            ax.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(integer=True)
            )

    # Label title and axes
    fig.suptitle(f"{spacing_matrix.name} Spacing")
    if spacing_matrix.n_strands == 2:
        axs[0, 0].set_ylabel("Forward")
        axs[1, 0].set_ylabel("Reverse")
        axs[1, 0].set_xlabel("Forward")
        axs[1, 1].set_xlabel("Reverse")

    # Draw colorbar
    fig.colorbar(heatmap, ax=axs, location="right")


def enrichment_plotter(
    ax: Axes,
    counts_obs: Tensor,
    counts_pred: Tensor,
    columns: list[int],
    kernel: int = 1,
    title: str = "",
    **kwargs: Any,
) -> (
    matplotlib.collections.PathCollection
    | matplotlib.collections.PolyCollection
):
    r"""Draws the enrichment scatterplot.

    Args:
        counts_obs: The observed counts in shape
            :math:`(\text{minibatch},\text{columns})`.
        counts_pred: The observed counts in shape
            :math:`(\text{minibatch},\text{columns})`.
        columns: The column indices to keep for plotting.
        kernel: The bin for average pooling of enrichment-sorted sequences.
        title: The plot title.
    """
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

    hexbin = len(columns) <= 2 and len(counts_pred) / kernel > 500

    # Plot binned enrichments
    n_bins = 0
    for i in range(len(columns) - 1):
        # Get the columns
        col1 = columns[i]
        col2 = columns[i + 1]
        cols_pred = counts_pred[:, [col1, col2]]
        cols_obs = counts_obs[:, [col1, col2]]

        # Remove rows which have 0's for both columns
        nonzero = torch.any(cols_obs > 0, dim=1)
        cols_pred, cols_obs = cols_pred[nonzero], cols_obs[nonzero]

        # Sort rows by predicted fold enrichment
        fold_enr = cols_pred[:, 1] / cols_pred[:, 0]
        sorting = torch.argsort(fold_enr, descending=True)
        cols_pred, cols_obs = cols_pred[sorting], cols_obs[sorting]

        # Bin the rows with AvgPool1D
        if kernel > 1:
            eps = 1 / kernel
            cols_pred = avg_pool1d(cols_pred, kernel)
            cols_obs = avg_pool1d(cols_obs, kernel)
        else:
            eps = 1
        n_bins += len(cols_obs)

        # Calculate enrichment
        x_pred = ((cols_pred[:, 1] + eps) / (cols_pred[:, 0] + eps)).float()
        y_obs = ((cols_obs[:, 1] + eps) / (cols_obs[:, 0] + eps)).float()
        fold_split = f"{(y_obs.max() / y_obs.min()).item():.2e}".split("e")
        coefficient = fold_split[0]
        power = fold_split[1]
        if power.startswith("-"):
            power = "-" + fold_split[1].lstrip("-0")
        else:
            power = fold_split[1].lstrip("+0")
        if len(power) == 0:
            power = "0"
        fold_range = rf"$\mathdefault{{{coefficient}\times10^{{{power}}}}}$"

        # Plot and print statistics
        spearman = scipy.stats.spearmanr(x_pred, y_obs).statistic
        pearson = scipy.stats.pearsonr(x_pred.log(), y_obs.log()).statistic
        rmsle = (x_pred.log() - y_obs.log()).square().mean().sqrt().item()
        label = f"{col1}→{col2}"
        label += (
            f" Obs. Enr. max / min = {fold_range}"
            f"\n$r_s$={spearman:.3f}, $r$={pearson:.3f}, RMSLE={rmsle:.3f}"
        )
        out: (
            matplotlib.collections.PathCollection
            | matplotlib.collections.PolyCollection
        )
        if hexbin:
            out = ax.hexbin(
                x_pred,
                y_obs,
                cmap=gnbu_mod,
                label=label,
                bins="log",
                xscale="log",
                yscale="log",
                **kwargs,
            )
        else:
            out = ax.scatter(x_pred, y_obs, label=label, alpha=0.5)
            ax.set_yscale("log")
            ax.set_xscale("log")

    # Make square
    min_range = 1.1 * min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_range = 0.9 * max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([min_range, max_range], [min_range, max_range], "k--")
    ax.set_xlim(0.9 * min_range, 1.1 * max_range)
    ax.set_ylim(0.9 * min_range, 1.1 * max_range)

    # Add title, legend, and labels
    if kernel > 1:
        title += f" ({n_bins:,} bins of n={kernel})"
    else:
        title += f" ({n_bins:,} probes)"
    ax.set_title(title)
    ax.set_xlabel("Predicted Enrichment")
    ax.set_ylabel("Observed Enrichment")
    ax.legend(loc="lower right")

    return out


def probe_enrichment(
    experiment: Experiment,
    batch: CountBatch,
    columns: list[int] | None = None,
    kernel: int = 500,
    max_split: int | None = None,
) -> None:
    """Plots the enrichment of sequences, binned by predicted enrichment.

    Args:
        experiment: An experiment modeling a selection.
        batch: A batch corresponding to the provided experiment.
        columns: The column indices to keep for plotting.
        kernel: The bin for average pooling of enrichment-sorted sequences.
        max_split: Maximum number of sequences scored at a time.
    """
    counts_obs, counts_pred = score(experiment, batch, max_split=max_split)
    if columns is None:
        columns = list(range(counts_obs.shape[1]))
    fig, ax = plt.subplots(figsize=(5, 5))
    out = enrichment_plotter(
        ax,
        counts_obs,
        torch.exp(counts_pred),
        columns=columns,
        kernel=kernel,
        title=f"{experiment.name} Probe-Level Enr.",
    )
    if isinstance(out, matplotlib.collections.PolyCollection):
        fig.colorbar(out, ax=ax)


def kmer_enrichment(
    experiment: Experiment,
    batch: CountBatch,
    columns: list[int] | None = None,
    kmer_length: int = 3,
    kernel: int = 500,
    max_split: int | None = None,
) -> None:
    """Plots the enrichment of k-mers, binned by predicted enrichment.

    Args:
        experiment: An experiment modeling a selection.
        batch: A batch corresponding to the provided experiment.
        columns: The column indices to keep for plotting.
        kernel: The bin for average pooling of enrichment-sorted k-mers.
        max_split: Maximum number of sequences scored at a time.
    """
    counts_obs, counts_pred = score(experiment, batch, max_split=max_split)
    kmer_counts, _ = count_kmers(batch.seqs, kmer_length=kmer_length)
    counts_obs = torch.tensor(kmer_counts @ counts_obs)
    counts_pred = torch.tensor(kmer_counts @ torch.exp(counts_pred))
    if columns is None:
        columns = list(range(counts_obs.shape[1]))
    fig, ax = plt.subplots(figsize=(5, 5))
    out = enrichment_plotter(
        ax,
        counts_obs,
        counts_pred,
        columns=columns,
        kernel=kernel,
        title=f"{experiment.name} {kmer_length}-mer Enr.",
    )
    if isinstance(out, matplotlib.collections.PolyCollection):
        fig.colorbar(out, ax=ax)


def kd_consistency(
    experiment: Experiment,
    i_index: int,
    b_index: int,
    u_index: int,
    batch: CountBatch,
    kernel: int = 500,
    max_split: int | None = None,
) -> None:
    """Plots the bound and unbound fractions, binned by predicted Kd.

    Args:
        experiment: An experiment modeling a Kd-seq selection.
        i_index: The index of the input round.
        b_index: The index of the BoundRound.
        u_index: The index of the UnboundRound.
        batch: A batch corresponding to the provided experiment.
        kernel: The bin for average pooling of Kd-sorted sequences.
        max_split: Maximum number of sequences scored at a time.
    """
    # Get Kd's
    i_round = experiment.rounds[i_index]
    b_round = experiment.rounds[b_index]
    u_round = experiment.rounds[u_index]
    free_protein = experiment.free_protein(i_index, b_index, u_index)
    k_a = torch.exp(
        score(b_round, batch, fun="log_aggregate", max_split=max_split)[1]
        - math.log(free_protein)
    )

    # Get counts of all rounds
    counts_obs, counts_pred = score(experiment, batch, max_split=max_split)
    counts_pred = torch.exp(counts_pred) * counts_obs.sum(dim=1, keepdim=True)

    # Sort by Ka
    sorting = torch.argsort(k_a)
    k_a = k_a[sorting]
    counts_obs = counts_obs[sorting]
    counts_pred = counts_pred[sorting]

    # Bin
    if kernel > 1:
        k_a = avg_pool1d(k_a, kernel)
        counts_obs = avg_pool1d(counts_obs, kernel)
        counts_pred = avg_pool1d(counts_pred, kernel)
        eps = 1 / kernel
    else:
        eps = 1.0

    # Calculate bound fractions
    pred_bound = (
        (counts_pred[:, b_index] + eps) / (counts_pred[:, i_index] + eps)
    ) * torch.exp(i_round.log_depth - b_round.log_depth).detach().cpu()
    obs_bound = (
        (counts_obs[:, b_index] + eps) / (counts_obs[:, i_index] + eps)
    ) * torch.exp(i_round.log_depth - b_round.log_depth).detach().cpu()

    # Calculate free fractions
    pred_unbound = (
        (counts_pred[:, u_index] + eps) / (counts_pred[:, i_index] + eps)
    ) * torch.exp(i_round.log_depth - u_round.log_depth).detach().cpu()
    obs_unbound = (
        (counts_obs[:, u_index] + eps) / (counts_obs[:, i_index] + eps)
    ) * torch.exp(i_round.log_depth - u_round.log_depth).detach().cpu()

    # Plot
    fig, ax = plt.subplots(
        2, 1, figsize=(5, 5), constrained_layout=True, sharex=True
    )
    axs = cast(AxesArray, ax)
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"Predicted $1/K_D$")

    rmse_bound = (pred_bound - obs_bound).square().mean().sqrt().item()
    axs[0].scatter(k_a, obs_bound, alpha=0.5, label=f"RMSE={rmse_bound:.3f}")
    axs[0].plot(k_a, pred_bound, "k--")
    axs[0].legend(loc="lower right")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Bound fraction")

    rmse_unbound = (pred_unbound - obs_unbound).square().mean().sqrt().item()
    axs[1].scatter(
        k_a, obs_unbound, alpha=0.5, label=f"RMSE={rmse_unbound:.3f}"
    )
    axs[1].plot(k_a, pred_unbound, "k--")
    axs[1].legend(loc="upper right")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("Unbound fraction")

    title = r"$K_D$ Model Consistency"
    if kernel > 1:
        title += f" ({len(counts_obs):,} bins of n={kernel})"
    else:
        title += f" ({len(counts_obs):,} probes)"
    fig.suptitle(title)


def keff_consistency(
    experiment: Experiment,
    batch: CountBatch,
    columns: list[int] | None = None,
    kernel: int = 500,
    max_split: int | None = None,
) -> None:
    """Plots the modified fraction, binned by predicted Kd.

    Args:
        experiment: An experiment modeling a Kinase-seq selection.
        batch: A batch corresponding to the provided experiment.
        columns: The column indices to keep for plotting.
        kernel: The bin for average pooling of Kd-sorted sequences.
        max_split: Maximum number of sequences scored at a time.
    """
    if columns is None:
        columns = [
            i
            for i, rnd in enumerate(experiment.rounds)
            if isinstance(rnd, ExponentialRound)
        ]

    _, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    axs = cast(Axes, ax)

    # Get counts of all rounds
    counts_obs, counts_pred = score(experiment, batch, max_split=max_split)
    counts_pred = torch.exp(counts_pred) * counts_obs.sum(dim=1, keepdim=True)

    n_bins = 0
    for e_col in columns:
        # Get rounds
        e_round = experiment.rounds[e_col]
        i_col = next(
            i
            for i, rnd in enumerate(experiment.rounds)
            if rnd is e_round.reference_round
        )
        i_round = experiment.rounds[i_col]

        # Get k_eff
        k_eff = torch.exp(
            score(e_round, batch, fun="log_aggregate", max_split=max_split)[1]
        )

        # Get counts
        cols_pred = counts_pred[:, [i_col, e_col]]
        cols_obs = counts_obs[:, [i_col, e_col]]

        # Remove rows with no counts
        nonzero = torch.any(cols_obs > 0, dim=1)
        cols_pred = cols_pred[nonzero]
        cols_obs = cols_obs[nonzero]
        k_eff = k_eff[nonzero]

        # Sort by k_eff
        sorting = torch.argsort(k_eff)
        k_eff = k_eff[sorting]
        cols_obs = cols_obs[sorting]
        cols_pred = cols_pred[sorting]

        # Bin
        if kernel > 1:
            k_eff = avg_pool1d(k_eff, kernel)
            cols_obs = avg_pool1d(cols_obs, kernel)
            cols_pred = avg_pool1d(cols_pred, kernel)
            eps = 1 / kernel
        else:
            eps = 1.0
        n_bins += len(cols_obs)

        # Calculate bound fractions
        pred_modified = (
            (cols_pred[:, 1] + eps) / (cols_pred[:, 0] + eps)
        ) * torch.exp(i_round.log_depth - e_round.log_depth).detach().cpu()
        obs_modified = (
            (cols_obs[:, 1] + eps) / (cols_obs[:, 0] + eps)
        ) * torch.exp(i_round.log_depth - e_round.log_depth).detach().cpu()

        # Plot
        rmse = (pred_modified - obs_modified).square().mean().sqrt().item()
        label = f"{e_round} RMSE={rmse:.3f}"
        axs.scatter(k_eff, obs_modified, alpha=0.5, label=label)
        axs.plot(k_eff, pred_modified, "k--")
        axs.set_xscale("log")
        axs.set_yscale("log")
        axs.set_xlabel(r"Predicted $k_{\mathrm{eff}}$")
        axs.set_ylabel("Modified fraction")
        axs.set_title(r"$k_{\mathrm{eff}}$ Model Consistency")
        axs.legend(loc="lower right")

    title = r"$k_{\mathrm{eff}}$ Model Consistency"
    if kernel > 1:
        title += f" ({n_bins:,} bins of n={kernel})"
    else:
        title += f" ({n_bins:,} probes)"
    axs.set_title(title)


def contribution(
    rnd: BaseRound | Aggregate,
    batch: CountBatch,
    kernel: int = 500,
    max_split: int | None = None,
) -> None:
    """Plots the predicted relative contribution of every Binding component.

    Args:
        rnd: A component containing an aggregate of different modes.
        batch: A batch corresponding to the provided experiment.
        kernel: The bin for average pooling of Kd-sorted sequences.
        max_split: Maximum number of sequences scored at a time.
    """
    _, log_aggregate = score(
        rnd,
        batch,
        fun="forward" if isinstance(rnd, Aggregate) else "log_aggregate",
        max_split=max_split,
    )

    bmd_names: list[str] = []
    bmd_contributions_list: list[Tensor] = []
    for ctrb in [m for m in rnd.modules() if isinstance(m, Contribution)]:
        bmd = ctrb.binding
        bmd_names.append("-".join(i.name for i in bmd.key()))
        bmd_contributions_list.append(
            score(ctrb, batch, max_split=max_split)[1].unsqueeze(1)
        )
    bmd_contributions = torch.cat(bmd_contributions_list, dim=1)

    fig, ax = plt.subplots(
        2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": (1, 4)}
    )
    axs = cast(AxesArray, ax)

    # Sort
    sorting = torch.argsort(log_aggregate, dim=0, descending=False)
    log_aggregate = log_aggregate[sorting]
    bmd_contributions = bmd_contributions[sorting]
    bmd_contributions -= bmd_contributions.logsumexp(dim=1, keepdim=True)

    # Bin
    bin_contributions = avg_pool1d(torch.exp(bmd_contributions), kernel)
    bin_partition = avg_pool1d(torch.exp(log_aggregate), kernel)
    x_min, x_max = torch.min(bin_partition), torch.max(bin_partition)

    # Stack plot
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

    # Density plot
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

    # Add title
    title = f"{rnd} Mode Contributions"
    if kernel > 1:
        title += f" ({len(bin_partition):,} bins of n={kernel})"
    else:
        title += f" ({len(bin_partition):,} probes)"
    axs[0].set_title(title)
    fig.align_ylabels(axs)
