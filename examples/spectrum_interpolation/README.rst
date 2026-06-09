Spectrum Interpolation Examples
===============================

Overview
--------

Examples demonstrating spectrum-interpolation line-noise removal
(Leske & Dalal, 2019). The power-line frequency and its harmonics are removed by
interpolating the amplitude spectrum across a narrow band while preserving the
phase, leaving broadband activity around the line frequency largely intact.

Files
-----

- ``plot_01_spectrum_interpolation.py``: Basic spectrum interpolation on
  synthetic data with 60 Hz line noise and harmonics.

Data Requirements
-----------------

- The example runs directly on synthetic data with no external downloads.

References
----------

- Leske, S., & Dalal, S. S. (2019). Reducing power line noise in EEG and MEG
  data via spectrum interpolation. NeuroImage, 189, 763-776.
