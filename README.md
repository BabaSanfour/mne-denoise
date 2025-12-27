# mne-denoise

`mne-denoise` provides denoising source separation (DSS) and narrow-band artefact
removal for electrophysiology data. It includes line noise suppression (ZapLine),
evoked response enhancement, and oscillatory signal extraction while preserving 
signal rank and interpretability.

The project follows the [MNE ecosystem](https://mne.tools/stable/about.html)
governance, coding standards, and community guidelines.

## Features

- **DSS (Denoising Source Separation)**: Linear and nonlinear spatial filtering
- **ZapLine / ZapLine-Plus**: Automatic line noise removal (50/60 Hz)
- **Narrowband DSS**: Extract oscillatory components (alpha, beta, etc.)
- **SSVEP DSS**: Enhance steady-state visually evoked potentials
- **Time-Shift DSS (TSR)**: Remove repetitive artifacts

## Installation

Stable releases are distributed on PyPI:

```bash
pip install mne-denoise
```

For development work, clone the repository and install in editable mode with the
documentation and testing extras:

```bash
pip install -e .[dev,docs]
```

## Quick Start

### ZapLine: Line Noise Removal

```python
import numpy as np
from mne_denoise.zapline import zapline_plus

# Simulated EEG with 50 Hz line noise
fs = 1000.0
t = np.arange(0, 10, 1 / fs)
line = 0.4 * np.sin(2 * np.pi * 50 * t)
data = np.random.randn(32, t.size) + line  # (n_channels, n_times)

result = zapline_plus(data, fs, noisefreqs='line')
clean_data = result.cleaned
```

### DSS: Denoising Source Separation

```python
import numpy as np
from mne_denoise.dss import compute_dss

# Data shape: (n_channels, n_times, n_trials)
data = np.random.randn(64, 1000, 100)
biased_data = data.mean(axis=2, keepdims=True)  # Trial average

filters, patterns, eigenvalues, _ = compute_dss(data, biased_data, n_components=10)
```

### Narrowband DSS: Extract Oscillatory Components

```python
from mne_denoise.dss import narrowband_dss

# Extract alpha band (8-12 Hz)
result = narrowband_dss(data, sfreq=500, freq_band=(8.0, 12.0), n_components=5)
```

## Documentation

TODO

## Roadmap

### ZapLine Integration
- [x] Core `dss_zapline` and `zapline_plus` functional API
- [x] `ZapLineResult` and `ZapLinePlusResult` dataclasses
- [ ] `ZapLine` transformer class (sklearn-compatible)
- [ ] Visualization utilities (PSD before/after plots)

### MNE Integration (Planned)

High-level functions to apply DSS/ZapLine directly to MNE objects:

```python
# Future API examples (not yet implemented)

# Apply DSS to MNE Raw
raw_clean = apply_dss_to_raw(raw, bias='alpha', n_components=10)

# Apply DSS to MNE Epochs  
epochs_clean = apply_dss_to_epochs(epochs, bias='evoked', n_components=5)

# Apply ZapLine to MNE Raw
raw_clean = apply_zapline_to_raw(raw, line_freq=50)

# Extract DSS components for visualization
components = get_dss_components(epochs, bias='evoked')
# Returns: filters, patterns, eigenvalues, sources, evoked_sources
```

### Other TODOs
- [ ] Complete documentation with Sphinx
- [ ] Add example notebooks
- [ ] Publish to PyPI
- [ ] Add pre-commit hooks and CI/CD

## Contributing

TODO

## License and Citation

This project is distributed under the terms of the [BSD 3-Clause
License](LICENSE). 

CITATION: TODO