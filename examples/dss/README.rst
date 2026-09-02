DSS examples
============

These examples show four scientifically distinct ways to define structure of
interest with Denoising Source Separation.

The evoked example uses trial reproducibility on real held-out MEG data. The
cardiac example uses event locking for artifact subtraction with an independent
clean-input preservation control. The narrowband example uses spectral
structure to recover a known target source. The TimeShiftDSS example extends
reproducibility into lag-augmented spatiotemporal space and uses held-out and
surrogate validation.

A DSS bias defines what the decomposition emphasizes; it does not by itself
establish that a selected component is neural signal or artifact.
