User Guide
==========
ProBound [#Rube2022]_ learns free energy parameters by maximizing the Poisson likelihood
given the count of each observed sequence across one or more sequential enrichment rounds.
The likelihood is computed by predicting the binding probability of a sequence,
and from this probability predicting the sequence count across the different enrichment rounds.

This User Guide covers all of the core features
designed into the original implementation of ProBound.
A simple example can be found in :doc:`CTCF: Single Experiment <_notebooks/CTCF>`.

Count Table
-----------

Alphabet
^^^^^^^^
The first step is to define the alphabet of the sequences being modeled

.. code-block:: python

    alphabet = pyprobound.alphabets.DNA()

Other than `DNA`, options include `RNA`, `Codon`, and `Protein`.
Custom alphabets can be used by creating a new instance of the class
:doc:`Alphabet <_autosummary/pyprobound.alphabets.Alphabet>`.
An example is :doc:`CEBPγ: EpiSELEX-seq <_notebooks/CEBPg>`.

Count Table
^^^^^^^^^^^
Each experiment must consist of one or more sequential enrichment rounds
that are sampled and sequenced in high-throughput.
The sequencing data must then be encoded into a count table
that enumerates the count of each sequence in each round.
A count table is a matrix :math:`k` such that :math:`k_{i,r}`
is the number of times that a probe :math:`i` appears in round :math:`r`.
For example,

===== ======= =======
Index Round 1 Round 2
===== ======= =======
ACGTC    2       1   
GAGTT    0       1   
GTCGC    1       2   
TCCAT    1       0   
===== ======= =======

Rounds 1 and 2 may correspond to successive SELEX rounds.
For `in-vivo` TF binding assays (such as ChIP-seq, ChIP-exo, CUT&Tag, CUT&RUN, ChIP-exo, etc.)
Round 1 would be the mock library, and Round 2 would be the assay library.

This count table should be loaded into a
`Pandas DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_,
where the index is the sequence label for that row.
If each round is stored as a single-column TSV, this can be done with a PyProBound helper function

.. code-block:: python

    dataframe = pyprobound.get_dataframe(["round1.tsv.gz", "round2.tsv.gz"])

Once loaded into a dataframe, it can be converted into a
:doc:`CountTable <_autosummary/pyprobound.table.CountTable>` object

.. code-block:: python

    count_table = pyprobound.CountTable(dataframe, alphabet=alphabet)

Binding Layer
-------------
In its simplest configuration, a binding model will consist of two distinct binding modes:
a non-specific binding parameter, and a position-specific affinity matrix (PSAM).

Non-Specific Binding
^^^^^^^^^^^^^^^^^^^^
:doc:`NonSpecific <_autosummary/pyprobound.layers.conv0d.NonSpecific>`
binding is a distinct binding mode, equivalent to a PSAM without any sequence-specific parameters

.. code-block:: python

    nonspecific = pyprobound.layers.NonSpecific(alphabet=alphabet, name="NS")

PSAM
^^^^
A :doc:`PSAM <_autosummary/pyprobound.layers.psam.PSAM>` represents a binding motif
as a matrix in which each element corresponds to the free-energy penalty
of binding for a given feature relative to a reference sequence that lacks that feature

.. math::

    \Delta\Delta G(\text{sequence}) = \Delta G(\text{sequence}) - \Delta G(\text{reference})

For example, the following creates a PSAM of length 16,
with a total of (4 bases)*(16 positions) = 64 parameters

.. code-block:: python

    psam = pyprobound.layers.PSAM(kernel_size=16, alphabet=alphabet)

PSAMs can be seeded with IUPAC code motifs, and can additionally model pairwise features
(such as dinucleotides, as well as non-adjacent letter pairs) and palindromic binding.
One example that uses all of these features is :doc:`CEBPγ: EpiSELEX-seq <_notebooks/CEBPg>`.
For further information, refer to the :doc:`PSAM API <_autosummary/pyprobound.layers.psam.PSAM>`.

Binding Mode
^^^^^^^^^^^^
Once the non-specific binding and PSAM objects are created, they must be wrapped
into a binding :doc:`Mode <_autosummary/pyprobound.mode.Mode>`.
For convenience, they will be wrapped into a single iterable

.. code-block:: python

    modes = [
        pyprobound.Mode.from_nonspecific(nonspecific, count_table),
        pyprobound.Mode.from_psam(psam, count_table),
    ]

Here, additional features can also be enabled, such as a position bias modeling,
which trains a multiplicative bias for each sliding window of the PSAM along
the sequence. One example of this is :doc:`GR: ChIP-seq <_notebooks/GR>`.
For further information, refer to the
`from_psam API <https://pyprobound.readthedocs.io/en/latest/_autosummary/pyprobound.mode.Mode.html#pyprobound.mode.Mode.from_psam>`_.

The output of each mode is :math:`-\log K^{rel}_{\text{D}}` of that mode, where

.. math::
    
    \frac{1}{K^{rel}_{\text{D}}(\text{sequence})}
    = \sum_{\text{window}} \frac{K_{\text{D}}(\text{reference})}{K_{\text{D}}(\text{window})}
    = \sum_{\text{window}} \exp \left( - \frac{\Delta\Delta G(\text{window})}{RT} \right)

where `window` is the sliding window of the PSAM along the sequence.
In computational terms, this is equivalent to the LogSumExp of the output
of a 1D convolution of the PSAM along the sequence.

Multiple Binding Modes
^^^^^^^^^^^^^^^^^^^^^^
When multiple binding modes are used to model an experiment,
ProBound learns an activity parameter :math:`\alpha_r` for each mode,
which estimates the ratio :math:`[P_{free}] / K_{\text{D}}(\text{reference})`
for a sequencing round :math:`r`.
Multiple binding modes can then be summed together analogously to the way
different sliding windows are summed together to predict :math:`K^{rel}_{\text{D}}`.
For a sequence :math:`i`, the sum over all modes becomes

.. math::

    Z_{i,r} = \sum_{\text{mode}} \frac{\alpha_{\text{mode}, r}}{K^{rel}_{\text{D, mode}}(\text{sequence})}

The value :math:`\log Z_{i,r}` can be calculated from
`log_aggregate <https://pyprobound.readthedocs.io/en/latest/_autosummary/pyprobound.rounds.BaseRound.html#pyprobound.rounds.BaseRound.log_aggregate>`_,
a function of a `Round` object described in the following section.

Assay Layer
-----------
To predict the count of a probe across different enrichment rounds,
the relationship between each round must first be encoded.

Specifically, if :math:`f_{i,r}` is the relative concentration
of probe :math:`i` in round :math:`r`, then the enrichment ratio
:math:`f_{i,r} / f_{i,r-1}` must be defined in terms of :math:`Z_{i,r}`.
A multiplicative sequencing depth parameter :math:`\eta_{r}` is also trained,
so the final output of a round is the expected log count of the probe in that round,
:math:`\log \left( \eta_r f_{i,r} \right)`, where

.. math::
    \eta_r f_{i,r} = \eta_r f_{i,r-1} \text{Enrichment}(Z_{i,r})

The assay layer is very flexible, so it must be carefully specified to
properly correspond to the experiment being modeled.

Initial Round
^^^^^^^^^^^^^
Each experiment begins with an :doc:`InitialRound <_autosummary/pyprobound.rounds.InitialRound>`

.. code-block:: python

    initial_round = pyprobound.rounds.InitialRound()

Subsequent Rounds
^^^^^^^^^^^^^^^^^
Each successive round can be described by an enrichment function relative to the preceding round.
For example, if a sample from the initial library is enriched for binding to a TF
to form the second round of a SELEX experiment, one can define

.. code-block:: python

    second_round = pyprobound.rounds.BoundRound.from_binding(modes, reference_round=initial_round)

If a third SELEX round was performed, then it could be created
with the flag `reference_round=second_round`, and so on.

:doc:`BoundRound <_autosummary/pyprobound.rounds.BoundRound>`
encodes the sigmoidal binding function as

.. math::
    f_{i,r} = f_{i,r-1} \frac{Z_{i,r}}{1 + Z_{i,r}}

Alternative enrichment functions can be specified,
such as unsaturated binding, catalytic enrichment, or modeling of the unbound fraction.
This can be used for Kinase-seq (see :doc:`Src: Kinase-seq <_notebooks/Src>`)
or Kd-seq (see :doc:`Dll: Kd-seq <_notebooks/Dll>`).
For further information, refer to the :doc:`rounds API <_autosummary/pyprobound.rounds>`.

Experiment
^^^^^^^^^^
Once all of the rounds are created, they can be combined into an
:doc:`Experiment <_autosummary/pyprobound.experiment.Experiment>`
in the order that they appear in the count table

.. code-block:: python

    experiment = pyprobound.Experiment([initial_round, second_round])

The output of an experiment is the output of the rounds,
normalized over the different rounds

.. math::
    \log \frac{\eta_{r} f_{i,r}}{
        \sum_{r^\prime} \eta_{r^\prime} f_{i, r^\prime}
    }

Sequencing Layer and Optimization
---------------------------------

Loss
^^^^
Multiple experiments can be trained through joint optimization. First, a
:doc:`MultiExperimentLoss <_autosummary/pyprobound.loss.MultiExperimentLoss>` must be created

.. code-block:: python

    model = pyprobound.MultiExperimentLoss([experiment])

The output of the model is the sum of the Poisson negative log-likelihoods of each experiment
(excluding constant terms),
normalized by the total number of observed sequences in their corresponding count tables.
Given a count table :math:`k`, this is

.. math::
    \frac{1}{\sum_{i,r} k_{i,r}}
    \sum_{i,r} k_{i,r} \log \frac{
        \eta_{r} f_{i,r}
    }{
        \sum_{r^\prime} \eta_{r^\prime} f_{i, r^\prime}
    }

The loss function may include different regularization penalties.
For further information, refer to the
:doc:`MultiExperimentLoss API <_autosummary/pyprobound.loss.MultiExperimentLoss>`.

Examples of jointly modeling multiple experiments with shared parameters
can be found in :doc:`CTCF: Multiple Experiments <_notebooks/CTCF_multiexp>`,
:doc:`UbxExdHth: Binding Cooperativity <_notebooks/UbxExdHth>`,
and :doc:`CEBPγ: EpiSELEX-seq <_notebooks/CEBPg>`.

Optimization
^^^^^^^^^^^^
To train the model, the model must then be wrapped into an
:doc:`Optimizer <_autosummary/pyprobound.optimizer.Optimizer>`

.. code-block:: python

    optimizer = pyprobound.Optimizer(
        model, [count_table], device="cpu", checkpoint="checkpoint.pt",
    )

The model can then be optimized using the optimization protocol
from the original ProBound publication with

.. code-block:: python

    optimizer.train_sequential()

The model will be saved to the file specified with the `checkpoint` keyword.
The output of the optimization can also be captured by specifying the `output` keyword.
Additional sampling, optimization, and early stopping parameters can also be provided.
One example that utilizes these features is
:doc:`Src: Kinase-seq with Early Stopping <_notebooks/Src_earlystop>`.
For further information, refer to the :doc:`Optimizer API <_autosummary/pyprobound.optimizer.Optimizer>`.

Additional Features
-------------------

Cooperativity
^^^^^^^^^^^^^
ProBound can also model the cooperativity between two transcription factors.
This is calculated as the product of the affinities of each factor at their respective binding sites,
multiplied by a bias trained for each relative distance between the two binding sites.
The relative affinity of the cooperative complex formed by factors A and B is

.. math::
    
    \frac{1}{K^{rel}_{\text{D, complex}}(\text{sequence})}
    = \sum_{\text{window A}} \sum_{\text{window B}} \frac{\omega_{A:B}(\text{window A}, \text{window B})}{K^{rel}_{\text{D, A}}(\text{window A}) K^{rel}_{\text{D, B}}(\text{window B})}

To train a cooperativity model, the bias parameter :math:`\omega_{A:B}`
must first be created from the two factors through a
:doc:`Spacing <_autosummary/pyprobound.cooperativity.Spacing>` object, which can then be wrapped
into a :doc:`Cooperativity <_autosummary/pyprobound.cooperativity.Cooperativity>` object.

.. code-block:: python

    spacing =  pyprobound.Spacing.from_specs([psam_A], [psam_B])
    cooperativity = pyprobound.Cooperativity(spacing, mode_A, modes_B)

The Cooperativity object can then be used just like a Mode object.
An example of cooperativity modeling can be found in
:doc:`UbxExdHth: Binding Cooperativity <_notebooks/UbxExdHth>`.

Kd-seq
^^^^^^
The ProBound publication [#Rube2022]_ describes the Kd-seq method, in which
the input, bound, and unbound libraries of an experiment are all sequenced and modeled jointly
to infer absolute binding constants.
An example is provided in :doc:`Dll: Kd-seq <_notebooks/Dll>`.

To implement Kd-seq in PyProBound, the input and bound libraries must be encoded
using the :doc:`InitialRound <_autosummary/pyprobound.rounds.InitialRound>` and
:doc:`BoundRound <_autosummary/pyprobound.rounds.BoundRound>` classes described in `Assay Layer`_.
For the bound round, the parameters :code:`library_concentration`
and :code:`target_concentration` must be specified in
`from_binding <https://pyprobound.readthedocs.io/en/latest/_autosummary/pyprobound.rounds.Round.html#pyprobound.rounds.Round.from_binding>`_.
These are the total concentrations of the DNA library and the TF, respectively.

.. code-block:: python

    initial_round = pyprobound.rounds.InitialRound()
    bound_round = pyprobound.rounds.BoundRound.from_binding(
        modes, initial_round, target_concentration=100, library_concentration=20
    )


Next, the unbound library, which encodes the complement of the sigmoidal binding function 

.. math::
    f_{i,r} = f_{i,r-1} \frac{1}{1 + Z_{i,r}}

must be specified. It can be created directly from the bound round with

.. code-block:: python

    unbound_round = pyprobound.rounds.UnboundRound.from_round(bound_round)

If the count table columns correspond to the input, bound, and unbound rounds, in that order,
the experiment can then be created with

.. code-block:: python

    experiment = pyprobound.Experiment([initial_round, bound_round, unbound_round])

Finally, after training, the
`free_protein <https://pyprobound.readthedocs.io/en/latest/_autosummary/pyprobound.experiment.Experiment.html#pyprobound.experiment.Experiment.free_protein>`_.
function of the :doc:`Experiment <_autosummary/pyprobound.experiment.Experiment>` object
can be used to calculate the free protein concentration.
The indices of the input, bound, and unbound rounds must be provided.
For example, if the experiment is defined as above, the function call would look like

.. code-block:: python

    free_protein = experiment.free_protein(0, 1, 2)

To calculate the free protein concentration in a different condition, such as
with a different DNA or TF concentration, the parameters :code:`target_concentration`
and :code:`library_concentration` can be passed separately to
`free_protein <https://pyprobound.readthedocs.io/en/latest/_autosummary/pyprobound.experiment.Experiment.html#pyprobound.experiment.Experiment.free_protein>`_.

Note that the units of :code:`target_concentration` and :code:`library_concentration`
must always be consistent, in both
`from_binding <https://pyprobound.readthedocs.io/en/latest/_autosummary/pyprobound.rounds.Round.html#pyprobound.rounds.Round.from_binding>`_
and `free_protein <https://pyprobound.readthedocs.io/en/latest/_autosummary/pyprobound.experiment.Experiment.html#pyprobound.experiment.Experiment.free_protein>`_.

Now What?
---------
Since parameters and outputs of these components are in terms of biophysical constants,
they can be used directly for interpreting experiments and validating against alternative assays.

Plotting
^^^^^^^^
PyProBound includes a plotting library, :code:`pyprobound.plotting`,
which must be imported separately. Several plotting functions
are provided in the :doc:`plotting API <_autosummary/pyprobound.plotting>`,
which are used throughout the Examples in the sidebar.

In :doc:`CTCF: Single Experiment <_notebooks/CTCF>`, examples include

* :doc:`pyprobound.plotting.logo <_autosummary/pyprobound.plotting.logo>`, which represents the PSAM as a sequence logo using `Logomaker <https://logomaker.readthedocs.io/>`_;
* :doc:`pyprobound.plotting.kmer_enrichment <_autosummary/pyprobound.plotting.kmer_enrichment>`, which plots the average enrichment of each subsequence of length `k` in the experiment;
* :doc:`pyprobound.plotting.probe_enrichment <_autosummary/pyprobound.plotting.kmer_enrichment>`, which plots the enrichment of each full sequence in the experiment and bins these values by the predicted value to overcome shot noise;
* and :doc:`pyprobound.plotting.contribution <_autosummary/pyprobound.plotting.contribution>`, which plots the contribution of each binding mode to the overall enrichment as a function of the level of enrichment, similarly binned as in `probe_enrichment`.

Validation
^^^^^^^^^^
To validate a model against a different dataset (for example, to evaluate
the performance of a SELEX-derived model on explaining MITOMI measurements),
one could directly use the output of a :code:`Mode` or :code:`Round` directly,
according to their biophysical definitions as described above.

There are, however, situations in which additional parameters need to be defined.
For example, if a protein binding microarray (PBM) experiment is used for validation,
the TF-DNA recognition model might be accurate at each individual binding site,
but there are often positional dependencies along the length of the probe
due to the design of the microarray [#Riley2015]_.

For this purpose, ProBound also contains a library, :code:`pyprobound.fitting`,
which must be imported separately. It allows for the retraining of experiment-dependent parameters,
such as positional dependencies in binding, while keeping experiment-independent parameters,
such as PSAM parameters, fixed.

Examples are provided at the bottom of :doc:`CEBPγ: EpiSELEX-seq <_notebooks/CEBPg>`,
:doc:`Dll: Kd-seq <_notebooks/Dll>`, and both :doc:`Src: Kinase-seq <_notebooks/Src>`
and :doc:`Src: Kinase-seq with Early Stopping <_notebooks/Src_earlystop>`.

There are two classes available in :code:`pyprobound.fitting`,
:doc:`Fit <_autosummary/pyprobound.fitting.Fit>`, which fits the function

.. math::
    \text{observation} (y) \sim m \times \text{prediction} (\log Z) + b

and :doc:`LogFit <_autosummary/pyprobound.fitting.LogFit>`, which fits the function

.. math::
    \log \left( \text{observation} (y) \right) \sim \log \left(
        \exp(m) \times \exp \left( \text{prediction} (\log Z) \right) + \exp(b)
    \right)

Here, :math:`y` is the observed value for each sequence
encoded as a single-column :doc:`CountTable <_autosummary/pyprobound.table.CountTable>`,
while `log Z` is the output of a :code:`log_aggregate` as described in `Multiple Binding Modes`_.

:math:`\text{prediction}` and :math:`\text{observation}` are callables passed by the user.
If not specified, :math:`\text{observation}` is the identity function by default.
:math:`\text{prediction}` must always be specified;
for example, if the observed value is proportional to binding, :math:`\text{prediction}` should be
:code:`F.sigmoid` and :code:`F.logsigmoid` for :code:`Fit` and :code:`LogFit`, respectively
(:code:`F` is a common alias for the `torch.nn.functional <https://pytorch.org/docs/stable/nn.functional.html>`_ library).

The constructors for :doc:`Fit <_autosummary/pyprobound.fitting.Fit>`
and :doc:`LogFit <_autosummary/pyprobound.fitting.LogFit>` contain many parameters.
The linear scaling factors :math:`m` and :math:`b` are trained only if :code:`train_offset=True`.
Additionally, :code:`update_construct=True`, which updates all experiment-specific parameters,
must be passed if the validation sequence length is different than the training sequence length.
If positional dependencies must be retrained, :code:`train_posbias=True` must also be provided.
In some cases, avidity may be captured with :code:`train_hill=True`.

Once the fitting object is created, the
`fit <https://pyprobound.readthedocs.io/en/latest/_autosummary/pyprobound.fitting.Fit.html#pyprobound.fitting.Fit.fit>`_
function can be used to train the linear scaling factors and experiment-specific parameters
and the `plot <https://pyprobound.readthedocs.io/en/latest/_autosummary/pyprobound.fitting.Fit.html#pyprobound.fitting.Fit.plot>`_
function can be used to plot how well the observed and expected values agree.

For further information, refer to the :doc:`fitting API <_autosummary/pyprobound.fitting>`.

Reference
---------
.. [#Rube2022] Rube, H.T., Rastogi, C., Feng, S. et al. Prediction of protein–ligand binding affinity from sequencing data with interpretable machine learning. Nat Biotechnol 40, 1520–1527 (2022). https://doi.org/10.1038/s41587-022-01307-0
.. [#Riley2015] Riley, T.R., Lazarovici, A., Mann, R.S., and Bussemaker, H.J. Building accurate sequence-to-affinity models from high-throughput in vitro protein-DNA binding data using FeatureREDUCE. eLife 4:e06397 (2015). https://doi.org/10.7554/eLife.06397 
