API Overview
============

PyProBound distinguishes between experiment-specific and experiment-independent parameters. For
example, the positional dependency of binding along the length of a sequence is experiment-specific,
whereas the sequence parameters in a PSAM should be independent of a specific experimental design.
The experiment-independent parameters are organized into a
:doc:`LayerSpec <_autosummary/pyprobound.layers.layer.LayerSpec>` object,
which is then wrapped into a :doc:`Layer <_autosummary/pyprobound.layers.layer.Layer>`
object that adds the experiment-specific parameters
and implements that layer's calculation in its forward function.

A single :doc:`Mode <_autosummary/pyprobound.mode.Mode>` can contain multiple :code:`Layer` s,
which are applied sequentially as in 
`torch.nn.Sequential <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_.

Each :doc:`Mode <_autosummary/pyprobound.mode.Mode>` is then joined with a
round-specific :code:`log_activity` parameter in
a :doc:`Contribution <_autosummary/pyprobound.aggregate.Contribution>` object.
Multiple :code:`Contribution` s can then be combined in an
:doc:`Aggregate <_autosummary/pyprobound.aggregate.Aggregate>` object,
which is used for calculating :math:`Z_{i,r}` as described in the
`User Guide <https://pyprobound.readthedocs.io/en/latest/UserGuide.html#multiple-binding-modes>`_.

An overview of the different classes and their attributes and types is shown below.

Additionally, most objects inherit from :doc:`Component <_autosummary/pyprobound.base.Component>`.
This object contains functions for checkpointing parameters to a file,
freezing and unfreezing parameters, implementing the sequential optimization procedure recursively,
and caching the output of a component to avoid recomputation if it appears multiple times in the
architecture of a model.

.. image:: _static/PyProBound.svg
  :alt: PyProBound class diagram
