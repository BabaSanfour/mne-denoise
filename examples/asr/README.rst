ASR Examples
============

Examples demonstrating Artifact Subspace Reconstruction for burst-artifact
repair in EEG data.

Files
-----

- ``plot_01_basic_usage.py``: Standard ASR on synthetic multichannel burst
  artifacts.
- ``plot_02_mne_raw_qc.py``: MNE ``Raw`` usage with repair annotations and
  optional final clean_windows-style rejection masks.
- ``plot_03_adaptive_asr.py``: AASR-style adaptive chunk updates with
  ``partial_fit()`` then ``transform()``.
- ``plot_04_juggler_asr.py``: Juggler-style DBSCAN calibration on dense short
  bursts.
- ``plot_05_asr_visualization.py``: The ``mne_denoise.viz`` ASR plotting
  helpers — overlay, cutoff sweep, variance topography, repair timeline, and a
  standard-vs-Riemannian method comparison.

Notes
-----

The examples use synthetic data so they can run without external downloads.
Apply ASR to real EEG only after bad-channel handling, referencing decisions,
and high-pass filtering have been made in the surrounding MNE workflow.
