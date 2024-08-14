Advanced Configuration
======================
Implementing ProBound in a PyTorch framework also allows for not only variable
length sequences, but also more flexible modeling approaches,
such as multi-layer binding modes and more complex assay configurations.
These will be covered in the context of modeling sequence-dependent bias
during the fragmentation step of ChIP-seq experiments as discussed in [#Li2023]_.
An example can be found in :doc:`CTCF: ChIP-seq <_notebooks/CTCF_ChIP-seq>`.


Multi-Round Enrichment
----------------------
Different `Round` classes can be used in the same experiment,
and different `Round` classes can contain different binding modes.
A single round can be used as a single reference for multiple rounds, such as in
Kd-seq where both the bound and unbound libraries are sampled from the same input library.

A `Round` does not even need to correspond to an observed sequencing round in the count table.
For example, the sonication step in a ChIP-seq experiment can be modeled
as an unobserved sequence-dependent bias round [#Li2023]_. For example,

.. code-block:: python

    round_initial = pyprobound.rounds.InitialRound()
    round_bias = pyprobound.rounds.BoundUnsaturatedRound.from_binding(
        bias_modes, round_initial,
    )
    round_bound = pyprobound.rounds.BoundUnsaturatedRound.from_binding(
        binding_modes, reference_round
    )
    experiment = pyprobound.Experiment([round_initial, round_bound])

Here, the observed enrichment is explained as a product of the :code:`bias_modes` and the
:code:`binding_modes`, even if there is no sequencing data available that isolates bias process.

Multi-Layer Modes
-----------------
A :doc:`Mode <_autosummary/pyprobound.mode.Mode>` can be made up of more than one
:doc:`Layer <_autosummary/pyprobound.layers.layer.Layer>`, each of which are applied sequentially.

For example, the sequence bias effect of sonication is observed at the ends of a fragment.
Therefore, the bias modes should only score the ends of the sequence.
This might be implemented by creating a :doc:`Roll <_autosummary/pyprobound.layers.roll.Roll>` layer
that indexes the 10 positions from the 3' end of a sequence (adjusting for variable lengths),
and then passing the output of that layer to the standard
:doc:`Conv1d <_autosummary/pyprobound.layers.conv1d.Conv1d>` layer used to score a
:doc:`PSAM <_autosummary/pyprobound.layers.psam.PSAM>`.

.. code-block:: python

    roll = pyprobound.layers.Roll.from_spec(
        pyprobound.layers.RollSpec(alphabet, direction="right", max_length=10),
        count_table,
    )
    conv1d = pyprobound.layers.Conv1d.from_psam(psam, roll_left)
    mode = pyprobound.Mode([roll, conv1d])

Even if a multi-layer model can't be encoded using PyProBound's
:doc:`Layers <_autosummary/pyprobound.layers.layer.Layer>`,
any `torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_
that takes one-hot encoded sequencing data of shape 
:math:`(\text{minibatch},\text{in_channels},\text{in_length})`
and returns scores of shape :math:`(\text{minibatch},1,1)` can be wrapped into a
:doc:`ModuleLayer <_autosummary/pyprobound.layers.module.ModuleLayer>`.

Modifying the Loss Function
---------------------------
While :doc:`MultiExperimentLoss <_autosummary/pyprobound.loss.MultiExperimentLoss>`
defines the Poisson negative log-likelihood of a model [#Rube2022]_,
a new Loss module can be created by implementing the abstract method
`negloglik <https://pyprobound.readthedocs.io/en/latest/_autosummary/pyprobound.loss.BaseLoss.html#pyprobound.loss.BaseLoss.negloglik>`_.

For example, :doc:`MultiRoundMSLELoss <_autosummary/pyprobound.loss.MultiRoundMSLELoss>`
implements the Mean Squared Logarithmic Error (MSLE). This can be combined with
position bias modeling and overhang binding padding to train a PSAM from Protein Binding microarray
(PBM) data, inspired by the FeatureREDUCE algorithm [#Riley2015]_.
An example can be found in :doc:`CEBPγ: PBM <_notebooks/CEBPg_PBM>`.

Other Changes
-------------
A ProBound model can also be imported into PyProBound using
:doc:`parse_probound_model <_autosummary/pyprobound.external.parse_probound_model>`.
This function changes the default settings of PyProBound components to reconcile the discrepancies
between ProBound and PyProBound, which include the following:

* PyProBound scores all PSAMs using the same flank length, so ProBound models
  that vary the flank length for each PSAM require padding the input of each
  Conv1d layer using :doc:`get_padding_layers <_autosummary/pyprobound.layers.pad.get_padding_layers>`.
* PyProBound always creates and regularizes parameters even if they not trained.
  For example, the position bias parameter will always be a component for every
  PSAM, so to import a ProBound model that does not contain a position bias
  parameter, the
  `exclude_regularization <https://pyprobound.readthedocs.io/en/latest/_autosummary/pyprobound.loss.MultiExperimentLoss.html#pyprobound.loss.MultiExperimentLoss.__init__>`_
  parameter must list that parameter.
* ProBound applies a constant non-specific activity parameter to all sequences.
  PyProBound, designed for variable-length sequences, scales this NS activity
  with the input sequence length, which makes the trained parameter smaller.
  ProBound can be replicated with :code:`NonSpecific(..., ignore_length=True)`.
* PyProBound scales regularization hyperparameters with the number of jointly
  trained experiments, so that likelihood and regularization terms maintain
  their relative weights. ProBound keeps hyperparameters fixed, which can be
  replicated with :code:`MultiExperimentLoss(..., dilute_regularization=True)`.
* PyProBound applies a weight to the loss of each experiment which by default
  is the inverse of the number of experiments. ProBound's calculation can be
  replicated with :code:`MultiExperimentLoss(..., weights=[1] * len(experiments))`.
* PyProBound utilizes an :doc:`InitialRound <_autosummary/pyprobound.rounds.InitialRound>`
  class for the input rounds, since its parameters cancel out when calculating predictions.
  ProBound treats input rounds the same as other rounds, which can be replicated by creating
  an input round as, for ex., :code:`BoundUnsaturatedRound(..., reference_round=None)`.

Troubleshooting
---------------
If the optimizer is not taking any optimization steps,
there are a couple of steps that can be taken to help it out.

1. When creating a Round object, increasing the default
   `activity_heuristic <https://pyprobound.readthedocs.io/en/latest/_autosummary/pyprobound.rounds.Round.html#pyprobound.rounds.Round.from_binding>`_
   parameter from 0.05 can help force the model to train a binding mode.
2. By default, PyProBound does not calculate the constant terms of the Poisson log-likelihood.
   However, including the constant terms can help with the numerics. This can be enabled with
   :code:`MultiExperimentLoss(..., full_loss=True)`.
3. The `optim_args <https://pyprobound.readthedocs.io/en/latest/_autosummary/pyprobound.optimizer.Optimizer.html#pyprobound.optimizer.Optimizer.__init__>`_
   dictionary is passed directly to the optimizer, which by default is
   `LBFGS <https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html>`_.
   For difficult optimization problems, increasing :code:`max_iter` or decreasing :code:`tolerance_grad` might help.

References
----------
.. [#Li2023] Li, X., Melo, L.A.N., and Bussemaker, H.J. Benchmarking DNA binding affinity models using allele-specific transcription factor binding data. bioRxiv (2023). https://doi.org/10.1038/s41587-022-01307-0
.. [#Rube2022] Rube, H.T., Rastogi, C., Feng, S. et al. Prediction of protein–ligand binding affinity from sequencing data with interpretable machine learning. Nat Biotechnol 40, 1520–1527 (2022). https://doi.org/10.1038/s41587-022-01307-0
.. [#Riley2015] Riley, T.R., Lazarovici, A., Mann, R.S., and Bussemaker, H.J. Building accurate sequence-to-affinity models from high-throughput in vitro protein-DNA binding data using FeatureREDUCE. eLife 4:e06397 (2015). https://doi.org/10.7554/eLife.06397 
