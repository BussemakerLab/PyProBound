PyProBound
==========

Implementation of ProBound [#Rube2022]_ in PyTorch.

ProBound is a method for learning free-energy parameters from enrichment-based assays.
While initially developed to learn relative affinities of TF-DNA binding from SELEX experiments,
it can also be configured to learn absolute binding affinities,
applied to `in-vivo` assays such as ChIP-seq or CUT&Tag,
and extended to profile the sequence-dependent kinetics of enzymes.

An overview of ProBound and PyProBound is provided in the :doc:`User Guide <UserGuide>`.

.. toctree::
   :hidden:
   :maxdepth: 3

   UserGuide
   AdvancedConfiguration

.. toctree::
   :caption: Examples
   :hidden:
   :titlesonly:

   CTCF
   CTCF_multiexp
   UbxExdHth
   CEBPg
   Dll
   GR
   Src
   Src_earlystop
   CEBPg_PBM
   CTCF_ChIP-seq

.. toctree::
   :caption: API
   :hidden:

   Overview

.. autosummary::
   :toctree: _autosummary
   :template: module.rst
   :recursive:

   pyprobound

Installation
------------
The PyPI package `pyProBound <https://pypi.org/project/pyprobound/>`_
is a Python wrapper for ProBoundTools, from the original Java implementation of ProBound.
The PyPI package is not maintained by the Bussemaker Lab.

To install the PyProBound package described in this documentation,
download directly from the `repository <https://github.com/BussemakerLab/PyProBound>`_ with

.. code-block::

   pip install git+https://github.com/BussemakerLab/PyProBound.git

References
----------
.. [#Rube2022] Rube, H.T., Rastogi, C., Feng, S. et al. Prediction of protein–ligand binding affinity from sequencing data with interpretable machine learning. Nat Biotechnol 40, 1520–1527 (2022). https://doi.org/10.1038/s41587-022-01307-0
