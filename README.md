# mne-denoise

`mne-denoise` provides narrow-band artefact removal tailored to MNE-Python
workflows. It wraps harmonic regression techniques to suppress power-line noise
and other oscillatory contaminants while preserving signal rank and
interpretability.

The project follows the [MNE ecosystem](https://mne.tools/stable/about.html)
governance, coding standards, and community guidelines.

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

## Quick start

```python
import numpy as np
from mne_denoise import zapline_plus

fs = 1000.0
t = np.arange(0, 10, 1 / fs)
line = 0.4 * np.sin(2 * np.pi * 50 * t)[:, None]
data = np.random.randn(t.size, 32) + line

clean, config, analytics, figs = zapline_plus(
    data,
    fs,
    noisefreqs="line",
    adaptiveNremove=True,
    plotResults=False,
)
```

All helpers accept MNE objects where appropriate. For example, denoise a Raw
recording in-place:

```python
import mne
from mne_denoise import apply_zapline_to_raw

raw = mne.io.read_raw_fif("mne_sample.fif", preload=True)
raw_clean, *_ = apply_zapline_to_raw(raw, line_freqs="line", plotResults=False)
raw_clean.plot()
```

## Documentation

TODO

## Contributing

TODO

## License and citation

This project is distributed under the terms of the [BSD 3-Clause
License](LICENSE). 

CITATION: TODO