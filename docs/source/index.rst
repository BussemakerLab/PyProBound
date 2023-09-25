PyProBound
==========

Implementation of ProBound [#Rube2022]_ in PyTorch.

.. toctree::
   :hidden:

   CTCF
   CTCF_multiexp
   UbxExdHth
   CEBPg
   Dll
   GR
   Src
   Src_earlystop

Examples
--------
- :doc:`CTCF: Single Experiment <_notebooks/CTCF>`
- :doc:`CTCF: Multiple Experiments <_notebooks/CTCF_multiexp>`
- :doc:`UbxExdHth: Binding Cooperativity <_notebooks/UbxExdHth>`
- :doc:`CEBPγ: EpiSELEX-seq <_notebooks/CEBPg>`
- :doc:`Dll: Kd-seq <_notebooks/Dll>`
- :doc:`GR: ChIP-seq <_notebooks/GR>`
- :doc:`Src: Kinase-seq <_notebooks/Src>`
- :doc:`Src: Kinase-seq with Early Stopping <_notebooks/Src_earlystop>`

Overview
--------
.. image:: _static/PyProBound.svg
  :alt: PyProBound class diagram

API
---
.. autosummary::
   :toctree: _autosummary
   :template: module.rst
   :recursive:

   pyprobound

Reference
----------
.. [#Rube2022] Rube, H.T., Rastogi, C., Feng, S. et al. Prediction of protein–ligand binding affinity from sequencing data with interpretable machine learning. Nat Biotechnol 40, 1520–1527 (2022). https://doi.org/10.1038/s41587-022-01307-0
