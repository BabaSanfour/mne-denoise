# mne-denoise

[![PyPI version](https://img.shields.io/pypi/v/mne-denoise.svg)](https://pypi.org/project/mne-denoise/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Denoising Source Separation (DSS) algorithms for MNE-Python.**

`mne-denoise` provides robust Denoising Source Separation (DSS) techniques to the Python MNE ecosystem. It specializes in extracting signals of interest (evoked responses, oscillations) by leveraging data structure (reproducibility across trials) rather than just variance.

## Features

-   **Linear DSS**: Isolate components based on reproducibility across trials (evoked response) or characteristic frequencies.
-   **Nonlinear / Iterative DSS**: Powerful separation for complex non-Gaussian sources.
-   **Spectrogram & Temporal DSS**: Specialized denoisers for time-frequency targets.
-   **MNE Integration**: Fit and transform directly on `mne.Raw`, `mne.Epochs`, and `mne.Evoked` objects.
-   **Scikit-Learn API**: Fully compatible `Estimator` interface (`fit`, `transform`, `inverse_transform`).

## Installation

Install via pip:

```bash
pip install mne-denoise
```

Or install from source:

```bash
pip install "git+https://github.com/mne-tools/mne-denoise.git"
```

## Quick Start

### Enhancing Evoked Responses (DSS)

DSS finds spatial filters that maximize the ratio of "reproducible" (evoked) power to total power.

```python
from mne_denoise.dss import DSS, TrialAverageBias
import mne

# Load epochs
epochs = mne.read_epochs("sample_epochs.fif")

# Define our "bias": we want components that are consistent across trials
bias = TrialAverageBias()

# Initialize and fit DSS
dss = DSS(bias=bias, n_components=5)
# Fits regular DSS on the epochs data
dss.fit(epochs)

# 1. Extract the clean components (sources)
sources = dss.transform(epochs, return_type="sources")

# 2. Or reconstruct sensor data using only the best components
cleaned_epochs = dss.transform(epochs, return_type="epochs")
```

### Narrowband DSS (Oscillations)

Extract specific rhythms (e.g. Alpha, Beta) by maximizing power in a frequency band.

```python
from mne_denoise.dss import BandpassBias

# Define a bandpass bias (8-12 Hz)
bias = BandpassBias(sfreq=epochs.info['sfreq'], freq=10, bandwidth=4)

dss_alpha = DSS(bias=bias, n_components=3)
dss_alpha.fit(epochs)

# Extract alpha components
alpha_sources = dss_alpha.transform(epochs, return_type="sources")
```

## Documentation

Full documentation is available at [https://mne-tools.github.io/mne-denoise/](https://mne-tools.github.io/mne-denoise/).

## Contributing

We welcome contributions! Please check [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## References

**DSS (Linear/Nonlinear):**
> Särelä, J., & Valpola, H. (2005). Denoising source separation. *Journal of Machine Learning Research*, 6, 233-272.
>
> de Cheveigné, A., & Simon, J. Z. (2008). Denoising based on spatial filtering. *Journal of Neuroscience Methods*, 171(2), 331-339.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.