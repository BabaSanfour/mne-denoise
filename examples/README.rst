Examples
========

The examples below show how ``mne-denoise`` methods can be used for specific
scientific denoising problems in EEG and MEG.

Each example starts from the structure that makes a method appropriate:
clean calibration data, spatial redundancy, a forward model, spectral
structure, repeated trials, or reference channels. Where possible, examples
evaluate both suppression of the target artifact and preservation of the
signal of interest.

Controlled simulations and deliberately contaminated recordings are used when
ground truth is needed. Real MNE datasets are used when sensor geometry,
recording structure, or container behavior is scientifically relevant.

The examples are demonstrations, not universal benchmarks or parameter
recommendations. Parameters should be validated for the recording and
scientific endpoint at hand.
