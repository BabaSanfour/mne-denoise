DSS Examples
============

Examples demonstrating Denoising Source Separation (DSS) for various use cases.

Basic DSS
---------
- ``plot_01_dss_fundamentals.py``: Core DSS concepts (Trial Average and Bandpass biases)
- ``plot_02_artifact_correction.py``: Removing artifacts using DSS
- ``plot_03_evoked_responses.py``: Enhancing event-related potentials

Spectral & Oscillatory DSS
--------------------------
- ``plot_04_spectral_dss.py``: Frequency-specific component extraction
- ``plot_05_periodic_dss.py``: SSVEP and periodic signal enhancement
- ``plot_06_temporal_dss.py``: Time-shift DSS for temporal structure
- ``plot_07_spectrogram_dss.py``: Time-frequency decomposition with DSS

Advanced DSS
------------
- ``plot_08_blind_source_separation.py``: Comparison with ICA
- ``plot_09_custom_bias.py``: Creating custom bias functions
- ``plot_10_benchmarking.py``: Performance evaluation
- ``plot_11_wiener_masking.py``: Nonlinear DSS with Wiener masks
- ``plot_12_joint_dss.py``: Multi-dataset repeatability (JDSS)

References
----------
- Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res.
- de Cheveigné & Simon (2008). Denoising based on spatial filtering. J. Neurosci. Methods.
- de Cheveigné & Parra (2014). Joint decorrelation. NeuroImage.
