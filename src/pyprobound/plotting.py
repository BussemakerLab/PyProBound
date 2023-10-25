"""Module of miscellaneous plotting functions."""
import copy
import math
import warnings
from collections.abc import Collection
from typing import Any, cast

import logomaker
import matplotlib
import matplotlib.axes
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
from .cooperativity import Cooperativity, Spacing
from .experiment import Experiment
from .layers import PSAM, Conv0d, Conv1d
from .rounds import BaseRound, ExponentialRound
from .table import CountBatch, score
from .utils import avg_pool1d

if "Arial" in matplotlib.font_manager.findfont("Arial"):
    matplotlib.rcParams["font.sans-serif"] = "Arial"
gnbu = plt.get_cmap("YlGnBu")(range(256))[64:]
gnbu_mod = matplotlib.colors.LinearSegmentedColormap.from_list(
    "gnbu_mod", gnbu
)
cmap = matplotlib.colormaps["bwr"].copy()
cmap.set_bad(color="gray")


def count_kmers(sequences: Collection[str], kmer_length: int = 3) -> Tensor:
    """Returns a sparse count matrix of k-mers in a list of sequences.

    Args:
        sequences: The sequences to count the k-mers in.
        kmer_length: The k-mer length to be counted.

    Returns:
        A Sparse CSC tensor of the count of each k-mer in each sequence.
    """
    vocabulary: dict[str, int] = {}
    data: list[int] = []
    indices: list[int] = []
    indptr: list[int] = [0]
    for seq in sequences:
        count: dict[int, int] = {}
        for view in range(0, len(seq) - kmer_length + 1):
            kmer = seq[view : view + kmer_length]

            # Get key from vocabulary
            if kmer not in vocabulary:
                key = len(vocabulary)
                vocabulary[kmer] = key
            else:
                key = vocabulary[kmer]

            # Running sum of kmers in sequence
            if key not in count:
                count[key] = 1
            else:
                count[key] += 1

        # Add to csc initializers
        indptr.append(indptr[-1] + len(count))
        data.extend(count.values())
        indices.extend(count.keys())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.sparse_csc_tensor(
            indptr,
            indices,
            data,
            size=(len(vocabulary), len(sequences)),
            dtype=torch.float32,
        )


def save_image(filename: str) -> None:
    """Saves an image to a matplotlib plot to a file."""
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def logo(
    psam: PSAM,
    reverse: bool = False,
    fix_gauge: bool = True,
    save: str | None = None,
) -> None:
    """Plots a sequence logo for the given PSAM using Logomaker.

    Args:
        psam: A PSAM to plot into a logo.
        reverse: Whether to plot the reverse complement.
        fix_gauge: Whether to call fix_gauge() before plotting the logo.
        save: The basename to write the plot to, if provided.
    """
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
            torch.movedim(psam.get_filter(i).detach(), -1, 1)[0]
            .float()
            .cpu()
            .numpy(),
        )
        for i in range(psam.pairwise_distance + 1)
    ]

    # Binding mode attributes
    in_channels = psam.in_channels
    size = len(psam.symmetry)
    positions = in_channels * np.arange(size) + (in_channels / 2) - 0.5

    # Set up subplots
    pairwise = psam.pairwise_distance > 0
    width = 7 if pairwise else 8
    logo_height = width / 4
    colorbar_width = width / 20
    if pairwise:
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

    # Set up monomer matrix
    matrix = matrices[0]
    matrix -= matrix.mean(1, keepdims=True)
    matrix = np.flip(matrix, axis=(0, 1)) if reverse else matrix

    if psam.alphabet is not None:
        # Create sequence logo
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
                ax=axs[0, 0] if pairwise else axs,
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
        axs_curr = axs[0, 0] if pairwise else axs
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

    # Create pairwise logo
    if pairwise:
        # Create empty heatmap matrix
        heatmap = np.empty((size * in_channels, size * in_channels))
        heatmap[:] = np.NaN

        # Fill heatmap matrix
        for dist in range(1, len(matrices)):
            matrix = matrices[dist]
            for pos in range(size - dist):
                x, y = (pos + dist) * in_channels, pos * in_channels
                heatmap[x : x + in_channels, y : y + in_channels] = matrix[pos]
                heatmap[y : y + in_channels, x : x + in_channels] = matrix[
                    pos
                ].T

        if reverse:
            heatmap = np.flipud(np.fliplr(heatmap))

        # Draw heatmap
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

        # Draw labels
        positions = in_channels * np.arange(size) + (in_channels / 2) - 0.5
        labels = np.arange(size)
        axs[1, 0].set_xticks(positions, labels)
        axs[1, 0].set_yticks(positions, labels)

        # Draw colorbar
        colorbar = plt.cm.ScalarMappable(cmap=cmap)
        colorbar.set_clim(-max_val, max_val)
        fig.colorbar(colorbar, ax=axs[1, 1], fraction=1, pad=0)

    # Add title
    title = psam.name
    if reverse:
        title += ", Reversed"
    if pairwise:
        fig.suptitle(title, y=1)
    else:
        plt.title(title)

    # Output
    plt.tight_layout()
    if isinstance(save, str):
        name = "".join(i for i in psam.name if i.isalnum())
        save_image(save + f"logo_{name}{'_rev' if reverse else ''}.png")
    else:
        plt.show()  # type: ignore[no-untyped-call]


def posbias(conv1d: Conv0d | Conv1d, save: str | None = None) -> None:
    r"""Plots the position bias profile :math:`\omega(x)`.

    Args:
        conv1d: A component containing a position bias profile.
        save: The basename to write the plot to, if provided.
    """
    for i_out, out in enumerate(
        conv1d.get_log_posbias().detach().float().cpu().unbind(1)
    ):
        if isinstance(conv1d, Conv0d) or conv1d.length_specific_bias:
            out = out[conv1d.min_input_length :]
        out = torch.exp(out)
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
        axs[0].set_title(f"{conv1d.layer_spec.name} posbias (O={i_out})")

        # Add labels
        axs[0].xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True)
        )
        axs[0].yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True)
        )
        positions = axs[0].get_xticks()
        step = int(positions[-1] - positions[-2])
        if step > 0 and (
            isinstance(conv1d, Conv0d) or conv1d.length_specific_bias
        ):
            positions = range(
                0, conv1d.max_input_length - conv1d.min_input_length + 1, step
            )
            labels = [conv1d.min_input_length + i for i in positions]
            axs[0].set_yticks(positions, labels)

        # Draw colorbar
        colorbar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(
            colorbar,
            ax=axs,
            fraction=1,
            pad=0,
            location="bottom" if rows < columns else "right",
        )

        # Output
        if isinstance(save, str):
            save_image(save + f"{conv1d}_O{i_out}_posbias.png")
        else:
            plt.show()  # type: ignore[no-untyped-call]


def cooperativity(
    spacing_matrix: Spacing | Cooperativity, save: str | None = None
) -> None:
    r"""Plots the cooperativity position bias :math:`\omega_{a:b}(x^a, x^b)`.

    Args:
        spacing_matrix: A component containing the cooperativity position bias.
        save: The basename to write the plot to, if provided.
    """
    fig = plt.figure(figsize=(6.4, 4.8), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, width_ratios=[1, 0.1])
    axs = subfigs[0].subplots(
        spacing_matrix.n_strands,
        spacing_matrix.n_strands,
        sharex=True,
        sharey=True,
        gridspec_kw={"right": 0.15},
    )

    # Get spacing matrix and axis labels
    if isinstance(spacing_matrix, Spacing):
        out = torch.exp(
            spacing_matrix.get_log_spacing(
                spacing_matrix.max_num_windows, spacing_matrix.max_num_windows
            )
            .detach()
            .float()
            .cpu()
        )
        cast(matplotlib.figure.SubFigure, subfigs[0]).supxlabel(
            "-".join([i.name for i in spacing_matrix.mode_key_a])
        )
        cast(matplotlib.figure.SubFigure, subfigs[0]).supylabel(
            "-".join([i.name for i in spacing_matrix.mode_key_b])
        )
    else:
        out = torch.exp(
            spacing_matrix.get_log_spacing().detach().cpu()
        ).squeeze()
        cast(matplotlib.figure.SubFigure, subfigs[0]).supxlabel(
            spacing_matrix.mode_a.name
        )
        cast(matplotlib.figure.SubFigure, subfigs[0]).supylabel(
            spacing_matrix.mode_b.name
        )

    # Draw subplots
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
            ax = cast(matplotlib.axes.Axes, axs[n_strand_0, n_strand_1])
            ax.imshow(strand_1, interpolation="none", cmap=cmap, norm=norm)
            ax.xaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(integer=True)
            )
            ax.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(integer=True)
            )

    # Label title and axes
    cast(matplotlib.figure.SubFigure, subfigs[0]).suptitle(
        f"{spacing_matrix.name} Spacing"
    )
    if spacing_matrix.n_strands == 2:
        cast(matplotlib.axes.Axes, axs[0, 0]).set_ylabel("Forward")
        cast(matplotlib.axes.Axes, axs[1, 0]).set_ylabel("Reverse")
        cast(matplotlib.axes.Axes, axs[1, 0]).set_xlabel("Forward")
        cast(matplotlib.axes.Axes, axs[1, 1]).set_xlabel("Reverse")

    # Draw colorbar
    bar_axs = subfigs[1].add_axes(
        [0, 0.143, 1, 0.795]  # left, bottom , width, height
    )
    bar_axs.axis("off")
    colorbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cast(matplotlib.figure.SubFigure, subfigs[1]).colorbar(
        colorbar, ax=bar_axs, fraction=1, pad=0
    )

    # Output
    if isinstance(save, str):
        save_image(save + f"{spacing_matrix}_spacing.png")
    else:
        plt.show()  # type: ignore[no-untyped-call]


def _enrichment(
    counts_obs: Tensor,
    counts_pred: Tensor,
    columns: list[int],
    kernel: int = 1,
    title: str = "",
) -> matplotlib.figure.Figure:
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

    # Setup subplots
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

    # Plot binned enrichments
    min_range = float("inf")
    max_range = float("-inf")
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

        # Update range
        min_range = min(min_range, x_pred.min().item(), y_obs.min().item())
        max_range = max(max_range, x_pred.max().item(), y_obs.max().item())

        # Plot and print statistics
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

    # Make square
    curr_ax = axs[0] if hexbin else axs
    curr_ax.plot([min_range, max_range], [min_range, max_range], "k--")
    curr_ax.set_xlim(0.9 * min_range, 1.1 * max_range)
    curr_ax.set_ylim(0.9 * min_range, 1.1 * max_range)

    # Add title, legend, labels, and colorbar
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
    batch: CountBatch,
    columns: list[int] | None = None,
    kernel: int = 500,
    max_split: int | None = None,
    save: str | None = None,
) -> None:
    """Plots the enrichment of sequences, binned by predicted enrichment.

    Args:
        experiment: An experiment modeling a selection.
        batch: A batch corresponding to the provided experiment.
        columns: The column indices to keep for plotting.
        kernel: The bin for average pooling of enrichment-sorted sequences.
        max_split: Maximum number of sequences scored at a time.
        save: The basename to write the plot to, if provided.
    """

    counts_obs, counts_pred = score(experiment, batch, max_split=max_split)
    counts_pred = torch.exp(counts_pred) * counts_obs.sum(dim=1, keepdim=True)
    if columns is None:
        columns = list(range(counts_obs.shape[1]))

    # Make plot
    _enrichment(
        counts_obs,
        counts_pred,
        columns=columns,
        kernel=kernel,
        title=f"{experiment.name} Probe-Level Enr.",
    )

    # Output
    if isinstance(save, str):
        name = "".join(i for i in experiment.name if i.isalnum())
        cols = "-".join(map(str, columns))
        save_image(save + f"probeenr_{name}_cols{cols}_kernel{kernel}.png")
    else:
        plt.show()  # type: ignore[no-untyped-call]


def kmer_enrichment(
    experiment: Experiment,
    batch: CountBatch,
    columns: list[int] | None = None,
    kmer_length: int = 3,
    kernel: int = 500,
    max_split: int | None = None,
    save: str | None = None,
) -> None:
    """Plots the enrichment of k-mers, binned by predicted enrichment.

    Args:
        experiment: An experiment modeling a selection.
        batch: A batch corresponding to the provided experiment.
        columns: The column indices to keep for plotting.
        kernel: The bin for average pooling of enrichment-sorted k-mers.
        max_split: Maximum number of sequences scored at a time.
        save: The basename to write the plot to, if provided.
    """

    counts_obs, counts_pred = score(experiment, batch, max_split=max_split)
    counts_pred = torch.exp(counts_pred) * counts_obs.sum(dim=1, keepdim=True)
    kmer_counts = count_kmers(batch.seqs, kmer_length=kmer_length)
    counts_obs = kmer_counts @ counts_obs
    counts_pred = kmer_counts @ counts_pred
    if columns is None:
        columns = list(range(counts_obs.shape[1]))

    # Make plot
    _enrichment(
        counts_obs,
        counts_pred,
        columns=columns,
        kernel=kernel,
        title=f"{experiment.name} {kmer_length}-mer Enr.",
    )

    # Output
    if isinstance(save, str):
        name = "".join(i for i in experiment.name if i.isalnum())
        cols = "-".join(map(str, columns))
        save_image(
            save + f"{kmer_length}merenr_{name}_cols{cols}_kernel{kernel}.png"
        )
    else:
        plt.show()  # type: ignore[no-untyped-call]


def kd_consistency(
    experiment: Experiment,
    i_index: int,
    b_index: int,
    u_index: int,
    batch: CountBatch,
    kernel: int = 500,
    max_split: int | None = None,
    save: str | None = None,
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
        save: The basename to write the plot to, if provided.
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

    # Output
    if isinstance(save, str):
        name = "".join(i for i in experiment.name if i.isalnum())
        columns = "-".join(str(i) for i in (i_index, b_index, u_index))
        save_image(save + f"kd_consistency_{name}_col{columns}.png")
    else:
        plt.show()  # type: ignore[no-untyped-call]


def keff_consistency(
    experiment: Experiment,
    batch: CountBatch,
    columns: list[int] | None = None,
    kernel: int = 500,
    max_split: int | None = None,
    save: str | None = None,
) -> None:
    """Plots the modified fraction, binned by predicted Kd.

    Args:
        experiment: An experiment modeling a Kinase-seq selection.
        batch: A batch corresponding to the provided experiment.
        columns: The column indices to keep for plotting.
        kernel: The bin for average pooling of Kd-sorted sequences.
        max_split: Maximum number of sequences scored at a time.
        save: The basename to write the plot to, if provided.
    """
    if columns is None:
        columns = [
            i
            for i, rnd in enumerate(experiment.rounds)
            if isinstance(rnd, ExponentialRound)
        ]

    _, axs = plt.subplots(figsize=(6, 3), constrained_layout=True)

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

    # Output
    if isinstance(save, str):
        name = "".join(i for i in experiment.name if i.isalnum())
        cols = "-".join(map(str, columns))
        save_image(save + f"kd_consistency_{name}_cols{cols}.png")
    else:
        plt.show()  # type: ignore[no-untyped-call]


def contribution(
    rnd: BaseRound | Aggregate,
    batch: CountBatch,
    kernel: int = 500,
    max_split: int | None = None,
    save: str | None = None,
) -> None:
    """Plots the predicted relative contribution of every Binding component.

    Args:
        rnd: A component containing an aggregate of different modes.
        batch: A batch corresponding to the provided experiment.
        kernel: The bin for average pooling of Kd-sorted sequences.
        max_split: Maximum number of sequences scored at a time.
        save: The basename to write the plot to, if provided.
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

    fig, axs = plt.subplots(
        2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": (1, 4)}
    )

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

    # Output
    if isinstance(save, str):
        name = "".join(i for i in rnd.name if i.isalnum())
        save_image(save + f"contribution_{name}_kernel{kernel}.png")
    else:
        plt.show()  # type: ignore[no-untyped-call]
