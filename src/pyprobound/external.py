"""Import models from ProBound, MotifCentral, JASPAR, and HOCOMOCO."""

import io
import json
import re
from collections.abc import Iterable, Sequence
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import requests
import torch
from Bio import motifs
from Bio.motifs import jaspar
from numpy.typing import NDArray
from pandas import DataFrame

from .aggregate import Aggregate, Contribution
from .alphabets import DNA, Alphabet, Protein
from .cooperativity import Cooperativity, Spacing
from .experiment import Experiment
from .layers import PSAM, Conv1d, Layer, NonSpecific, get_padding_layers
from .loss import MultiExperimentLoss
from .mode import Mode
from .rounds import (
    BoundRound,
    BoundUnsaturatedRound,
    ExponentialRound,
    RhoGammaRound,
    Round,
)
from .table import CountTable

FloatArray: TypeAlias = NDArray[Any]


def get_fit_final(path: str) -> dict[str, Any]:
    """Gets the best model fit from the fit.models.json output by ProBound."""
    fits = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            fits.append(json.loads(line))
    return min(fits, key=lambda fit: fit["metadata"]["logLikelihood"])


def parse_probound_alphabet(fit_final: dict[str, Any]) -> Alphabet:
    """Parses the alphabet used in a ProBound fit into an Alphabet."""
    model_settings = fit_final["modelSettings"]
    if (
        "letterOrder" not in model_settings
        or model_settings["letterOrder"] == "ACGT"
    ):
        return DNA()
    if model_settings["letterOrder"] == Protein().alphabet:
        return Protein()
    if "letterComplement" in model_settings:
        letters = ""
        for pair in model_settings["letterComplement"].split(","):
            letters = pair[0] + letters + pair[-1]
        return Alphabet(letters, complement=True)
    return Alphabet(model_settings["letterOrder"], complement=True)


def parse_probound_tables(
    fit_final: dict[str, Any], dataframes: Iterable[DataFrame]
) -> list[CountTable]:
    """Parses the count tables used in a ProBound fit into CountTables.

    Args:
        fit_final: The ProBound fit, parsed by get_fit_final.
        dataframes: The raw count tables, parsed by get_dataframe.

    Returns:
        A list of CountTable objects.
    """
    model_settings = fit_final["modelSettings"]
    alphabet = parse_probound_alphabet(fit_final)
    max_flank_length = max(
        i["flankLength"] for i in model_settings["bindingModes"]
    )
    count_tables = []
    for dataframe, table_settings in zip(
        dataframes, model_settings["countTable"], strict=True
    ):
        transliterate = dict(
            zip(
                table_settings["transliterate"]["in"],
                table_settings["transliterate"]["out"],
            )
        )
        count_tables.append(
            CountTable(
                dataframe[[i + 1 for i in table_settings["modeledColumns"]]],
                alphabet,
                transliterate=transliterate,
                transliterate_flanks=False,
                left_flank=table_settings["leftFlank"],
                right_flank=table_settings["rightFlank"],
                left_flank_length=min(
                    max_flank_length, len(table_settings["leftFlank"])
                ),
                right_flank_length=min(
                    max_flank_length, len(table_settings["rightFlank"])
                ),
            )
        )

    return count_tables


def parse_probound_psam(fit_final: dict[str, Any], mode_idx: int) -> PSAM:
    """Parses a single PSAM from a ProBound fit.

    Args:
        fit_final: The ProBound fit, parsed by get_fit_final.
        mode_idx: The index of the binding mode in the fit.

    Returns:
        A PSAM.
    """
    alphabet = parse_probound_alphabet(fit_final)
    alphalen = len(alphabet.alphabet)
    letter_order = {
        alphabet.get_index[val]: i
        for i, val in enumerate(fit_final["modelSettings"]["letterOrder"])
    }
    pairwise_order = {
        ori_1 * alphalen + ori_2: new_1 * alphalen + new_2
        for ori_1, new_1 in letter_order.items()
        for ori_2, new_2 in letter_order.items()
    }
    mode_settings = fit_final["modelSettings"]["bindingModes"][mode_idx]
    mode_coefficients = fit_final["coefficients"]["bindingModes"][mode_idx]
    psam = PSAM(
        kernel_size=mode_settings["size"],
        alphabet=alphabet,
        pairwise_distance=min(
            mode_settings["dinucleotideDistance"], mode_settings["size"] - 1
        ),
        score_reverse=not mode_settings["singleStrand"],
    )

    for key, param in psam.betas.items():
        elements = [int(i) for i in key.split("-")]
        elements = [i - 1 for i in elements[:-1]] + elements[-1:]
        if len(elements) == 2:
            torch.nn.init.constant_(
                param,
                mode_coefficients["mononucleotide"][
                    elements[0] * len(alphabet.alphabet)
                    + letter_order[elements[1]]
                ],
            )
        else:
            torch.nn.init.constant_(
                param,
                mode_coefficients["dinucleotide"][
                    elements[1] - elements[0] - 1
                ][
                    elements[0] * (len(alphabet.alphabet) ** 2)
                    + pairwise_order[elements[2]]
                ],
            )

    return psam


def parse_probound_modes(
    fit_final: dict[str, Any],
    layer_specs: Iterable[NonSpecific | PSAM],
    count_table: CountTable,
    table_idx: int,
) -> list[Mode]:
    """Parses all modes associated with an experiment in a ProBound fit.

    Args:
        fit_final: The ProBound fit, parsed by get_fit_final.
        layer_specs: The NonSpecific or PSAM encoding of the binding modes.
        count_table: The CountTable of that experiment, from
            parse_probound_tables.
        table_idx: The index of the experiment in the ProBound fit.

    Returns:
        A list of Mode objects.
    """
    alphabet = parse_probound_alphabet(fit_final)
    modes = []
    for mode_idx, layer_spec in enumerate(layer_specs):
        mode_settings = fit_final["modelSettings"]["bindingModes"][mode_idx]
        posbias = mode_settings["positionBias"]
        if isinstance(layer_spec, NonSpecific):
            modes.append(Mode.from_nonspecific(layer_spec, count_table))
        else:
            # Get padding
            flank_length = mode_settings["flankLength"]
            prev: Layer | CountTable
            if (
                flank_length < count_table.left_flank_length
                or flank_length < count_table.right_flank_length
            ):
                layers = get_padding_layers(
                    alphabet,
                    count_table,
                    min(0, flank_length - count_table.left_flank_length),
                    min(0, flank_length - count_table.right_flank_length),
                )
                prev = layers[-1]
            else:
                layers = []
                prev = count_table

            # Create mode
            layers.append(
                Conv1d.from_psam(layer_spec, prev, train_posbias=posbias)
            )
            mode = Mode(layers)

            # Import position bias parameters
            if posbias:
                for out_idx, _ in enumerate(mode.layers[-1].log_posbias[0]):
                    mode.layers[-1].log_posbias[0, out_idx] = torch.Tensor(
                        fit_final["coefficients"]["bindingModes"][mode_idx][
                            "positionBias"
                        ][table_idx][out_idx][:: 1 if out_idx == 0 else -1]
                    )

            modes.append(mode)

    return modes


def parse_probound_interactions(
    fit_final: dict[str, Any], modes: Sequence[Mode], table_idx: int
) -> list[Cooperativity]:
    """Parses all interactions associated with an experiment in a ProBound fit.

    Args:
        fit_final: The ProBound fit, parsed by get_fit_final.
        modes: The Mode encoding of binding modes, from parse_probound_modes.
        table_idx: The index of the experiment in the ProBound fit.

    Returns:
        A list of Mode objects.
    """
    cooperativities = []
    for coop_settings, coop_coefficients in zip(
        fit_final["modelSettings"]["bindingModeInteractions"],
        fit_final["coefficients"]["bindingModeInteractions"],
    ):
        # Get parameters
        max_spacing = None
        max_overlap = None
        if coop_settings["maxSpacing"] != -1:
            max_spacing = coop_settings["maxSpacing"]
        if coop_settings["maxOverlap"] != -1:
            max_overlap = coop_settings["maxOverlap"]

        # Create cooperativity
        mode_a = modes[coop_settings["bindingModes"][0]]
        mode_b = modes[coop_settings["bindingModes"][1]]
        cooperativity = Cooperativity(
            Spacing(
                mode_a.key(),
                mode_b.key(),
                max_overlap=max_overlap,
                max_spacing=max_spacing,
                ignore_pad=True,
            ),
            mode_a,
            mode_b,
        )
        cooperativities.append(cooperativity)

        # Parse log_posbias
        log_posbias = torch.tensor(
            coop_coefficients["positionMatrix"][table_idx]
        )
        log_posbias_shape = cooperativity.get_log_spacing_matrix()[0, 0].shape
        log_posbias = log_posbias[
            :, :, : log_posbias_shape[-2], : log_posbias_shape[-1]
        ]  # Trim if necessary
        assert log_posbias.shape == log_posbias_shape
        log_posbias[1] = log_posbias[1].flip(-2)  # ProBound writes 5'->3'
        log_posbias[:, 1] = log_posbias[:, 1].flip(-1)

        # Fill in log_posbias values
        for strand_a in range(log_posbias_shape[0]):
            for strand_b in range(log_posbias_shape[1]):
                square = log_posbias[strand_a, strand_b]
                for posbias_idx, diag_idx in enumerate(
                    range(-square.shape[0] + 1, square.shape[1])
                ):
                    cooperativity.log_posbias[posbias_idx][
                        0, 0, strand_a, strand_b
                    ] = square.diagonal(diag_idx)

    return cooperativities


def parse_probound_aggregate(
    fit_final: dict[str, Any],
    modes: Iterable[Mode],
    cooperativities: Iterable[Cooperativity],
    table_idx: int,
    round_idx: int,
) -> Aggregate:
    """Parses all binding modes and interactions of a round into an Aggregate.

    Args:
        fit_final: The ProBound fit, parsed by get_fit_final.
        modes: The Mode encoding of binding modes, from parse_probound_modes.
        cooperativities: The Mode encoding of binding mode interactions, from
            parse_probound_interactions.
        table_idx: The index of the experiment in the ProBound fit.
        round_idx: The index of the sequencing round in the experiment.

    Returns:
        An Aggregate.
    """
    enrichment_settings = fit_final["modelSettings"]["enrichmentModel"][
        table_idx
    ]
    contributions: list[Contribution] = []
    throwaway_activities: list[float] = []
    for mode_idx, mode in enumerate(modes):
        log_activity = fit_final["coefficients"]["bindingModes"][mode_idx][
            "activity"
        ][table_idx][round_idx]
        if mode_idx in enrichment_settings["bindingModes"]:
            contributions.append(
                Contribution(binding=mode, log_activity=log_activity)
            )
        else:
            throwaway_activities.append(log_activity)
    for coop_idx, coop in enumerate(cooperativities):
        log_activity = fit_final["coefficients"]["bindingModeInteractions"][
            coop_idx
        ]["activity"][table_idx][round_idx]
        if coop_idx in enrichment_settings["bindingModeInteractions"]:
            contributions.append(
                Contribution(binding=coop, log_activity=log_activity)
            )
        else:
            throwaway_activities.append(log_activity)
    aggregate = Aggregate(
        contributions=contributions,
        target_concentration=enrichment_settings["concentration"],
    )
    aggregate.throaway_activities = torch.nn.ParameterList(
        torch.tensor(log_activity) for log_activity in throwaway_activities
    )
    return aggregate


def parse_probound_rounds(
    fit_final: dict[str, Any],
    modes: Iterable[Mode],
    cooperativities: Iterable[Cooperativity],
    table_idx: int,
) -> list[Round]:
    """Parses all rounds associated with an experiment in a ProBound fit.

    Args:
        fit_final: The ProBound fit, parsed by get_fit_final.
        modes: The Mode encoding of binding modes, from parse_probound_modes.
        cooperativities: The Mode encoding of binding mode interactions, from
            parse_probound_interactions.
        table_idx: The index of the experiment in the ProBound fit.

    Returns:
        A list of Round objects.
    """
    enrichment_settings = fit_final["modelSettings"]["enrichmentModel"][
        table_idx
    ]
    rounds: list[Round] = []
    log_depths = fit_final["coefficients"]["countTable"][table_idx]["h"]
    for round_idx, log_depth in enumerate(log_depths):
        # Get round type
        constructor: (
            type[BoundRound]
            | type[BoundUnsaturatedRound]
            | type[RhoGammaRound]
            | type[ExponentialRound]
        )
        if enrichment_settings["modelType"] == "SELEX":
            constructor = (
                BoundRound
                if enrichment_settings["bindingSaturation"]
                else BoundUnsaturatedRound
            )
        elif enrichment_settings["modelType"] == "RhoGamma":
            constructor = RhoGammaRound
        elif enrichment_settings["modelType"] == "ExponentialKinetics":
            constructor = ExponentialRound
        else:
            raise RuntimeError(
                f"modelType {enrichment_settings['modelType']} not recognized"
            )

        # Create next round
        if enrichment_settings["cumulativeEnrichment"]:
            reference_round = rounds[-1] if len(rounds) > 0 else None
        else:
            reference_round = rounds[0] if len(rounds) > 0 else None
        next_round = constructor(
            aggregate=parse_probound_aggregate(
                fit_final, modes, cooperativities, table_idx, round_idx
            ),
            reference_round=reference_round,
            log_depth=log_depth,
            train_depth=False,
        )

        # Fill in remaining parameters
        enrichment_coefficients = fit_final["coefficients"]["enrichmentModel"][
            table_idx
        ]
        if enrichment_settings["modelType"] == "RhoGamma":
            torch.nn.init.constant_(
                next_round.rho, enrichment_coefficients["rho"][round_idx]
            )
            torch.nn.init.constant_(
                next_round.gamma, enrichment_coefficients["gamma"][round_idx]
            )
        if enrichment_settings["modelType"] == "ExponentialKinetics":
            torch.nn.init.constant_(
                next_round.delta, enrichment_coefficients["delta"][round_idx]
            )

        rounds.append(next_round)

    return rounds


def parse_probound_model(
    fit_final: dict[str, Any],
    config: dict[str, Any],
    count_tables: Iterable[CountTable],
) -> MultiExperimentLoss:
    """Parses all rounds associated with an experiment in a ProBound fit.

    Args:
        fit_final: The ProBound fit, parsed by get_fit_final.
        config: The config.json file, parsed into a dictionary.
        count_tables: The CountTable objects of an experiment, from
            parse_probound_tables.

    Returns:
        A MultiExperimentLoss.
    """
    # Parse alphabet
    alphabet = parse_probound_alphabet(fit_final)

    # Parse PSAMs
    layer_specs: list[NonSpecific | PSAM] = []
    for mode_idx, mode_settings in enumerate(
        fit_final["modelSettings"]["bindingModes"]
    ):
        if mode_settings["size"] == 0:
            layer_specs.append(
                NonSpecific(alphabet=alphabet, ignore_length=True)
            )
        else:
            layer_specs.append(parse_probound_psam(fit_final, mode_idx))

    # Parse experiments
    experiments: list[Experiment] = []
    for table_idx, count_table in enumerate(count_tables):
        modes = parse_probound_modes(
            fit_final, layer_specs, count_table, table_idx
        )
        cooperativities = parse_probound_interactions(
            fit_final, modes, table_idx
        )
        rounds = parse_probound_rounds(
            fit_final, modes, cooperativities, table_idx
        )
        experiments.append(
            Experiment(rounds, counts_per_round=count_table.counts_per_round)
        )

    # Get which parameters to exclude from regularization
    exclude_regularization = [
        "log_target_concentration",
        "log_hill",
        "layer_spec.bias",
        "log_spacing",
    ]
    for expt_idx, expt in enumerate(experiments):
        for ctrb_idx, ctrb in enumerate(
            expt.rounds[0].aggregate.contributions
        ):
            if (
                isinstance(ctrb.binding, Mode)
                and not ctrb.binding.layers[-1].train_posbias
            ):
                exclude_regularization.append(
                    f"experiments.{expt_idx}.rounds.0.aggregate"
                    f".contributions.{ctrb_idx}.binding.layers"
                    f".{len(ctrb.binding.layers)-1}.log_posbias"
                )

    return MultiExperimentLoss(
        experiments,
        lambda_l2=config["optimizerSetting"]["lambdaL2"],
        pseudocount=config["optimizerSetting"]["pseudocount"],
        exponential_bound=config["optimizerSetting"]["expBound"],
        full_loss=False,
        dilute_regularization=True,
        weights=[1] * len(experiments),
        exclude_regularization=exclude_regularization,
    )


def import_motif_central(fit_id: int) -> PSAM:
    """Parses a PSAM from a fit_id in MotifCentral."""
    model = json.loads(
        requests.get(
            "https://prod-gateway.motifcentral.org/"
            f"cellx/api/web/utility/fit/{fit_id}",
            timeout=5,
        ).text
    )
    return parse_probound_psam(model, -1)


def import_matrix(matrix: FloatArray, alphabet: Alphabet) -> PSAM:
    """Parses a PSAM from a (alphabet size, length) matrix."""
    matrix = np.asarray(matrix)
    if len(matrix) != len(alphabet.alphabet):
        raise ValueError(
            f"Second dimension of {matrix}"
            f" does not match size of alphabet {alphabet}"
        )
    psam = PSAM(kernel_size=len(matrix[0]), alphabet=alphabet, normalize=False)

    for key, param in psam.betas.items():
        elements = [int(i) for i in key.split("-")]
        elements = [i - 1 for i in elements[:-1]] + elements[-1:]
        if len(elements) == 2:
            torch.nn.init.constant_(param, matrix[elements[1]][elements[0]])

    return psam


def import_hocomoco(model: str) -> PSAM:
    """Parses a PSAM from a model in HOCOMOCO 11."""
    root = "https://hocomoco11.autosome.org"
    response = requests.get(f"{root}/motif/{model}", timeout=5)
    match = re.search(r"/final_bundle/.*\.pwm", response.text)
    if match is None:
        raise RuntimeError(f"Could not find HOCOMOCO PWM for {model}")
    text = requests.get(f"{root}/{match.group(0)}", timeout=5).text
    matrix = pd.read_csv(
        io.StringIO(text), sep="\t", header=None, skiprows=1
    ).to_numpy()
    matrix = matrix.T
    matrix -= matrix.max(axis=0, keepdims=True)
    return import_matrix(matrix, DNA())


def import_jaspar(fit_id: str) -> PSAM:
    """Parses a PSAM from an ID in JASPAR."""
    text = requests.get(
        f"https://jaspar.elixir.no/api/v1/matrix/{fit_id}.jaspar", timeout=5
    ).text
    motif = motifs.read(io.StringIO(text), "jaspar")  # type: ignore[no-untyped-call]
    motif.pseudocounts = jaspar.calculate_pseudocounts(motif)  # type: ignore[no-untyped-call]
    matrix = np.array(list(motif.pssm.values())) / np.log2(np.e)
    matrix -= matrix.max(axis=0, keepdims=True)
    return import_matrix(matrix, DNA())
