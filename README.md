# PyZaplinePlus

The package focuses on removing narrow-band artefacts such as power-line noise
from multichannel recordings (EEG, MEG, EMG, …) using harmonic regression and a
handful of lightweight heuristics.

## Installation

```bash
pip install -e .
```

The code depends on `numpy`, `scipy`, and `matplotlib`. The optional MNE helper
also requires `mne`. All dependencies are widely available on PyPI.

## Quick start

```python
import numpy as np
from pyzaplineplus import zapline_plus

# Fake signal: 32-channel noise plus a 50 Hz line component
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

`zapline_plus` returns four values:

1. the cleaned data array,
2. the effective configuration (dict),
3. an analytics dictionary describing each processed frequency, and
4. a list of matplotlib figures (empty unless plotting is enabled).

## Using the class interface

```python
from pyzaplineplus import PyZaplinePlus

runner = PyZaplinePlus(data, fs, noisefreqs=[50])
clean, config, analytics, figs = runner.run()
```

The object stores the prepared data and can be reused with updated parameters
between runs.

## Working with MNE

```python
import mne
from pyzaplineplus import apply_zapline_to_raw

raw = mne.io.read_raw_fif("mne_sample.fif", preload=True)
raw_clean, *_ = apply_zapline_to_raw(raw, line_freqs="line", plotResults=False)
raw_clean.plot()
```

## Configuration overview

Only a subset of the most relevant options is listed below. See
`pyzaplineplus/core.py` for the full catalogue and inline documentation.

| Key | Meaning | Default |
| --- | --- | --- |
| `noisefreqs` | frequency list, `"line"`, or empty for auto detection | `[]` |
| `minfreq` / `maxfreq` | search bounds for automatic detection | `17` / `99` |
| `detectionWinsize` | coarse window width in Hz | `6` |
| `chunkLength` | fixed chunk length in seconds (`0` disables splitting) | `0` |
| `minChunkLength` | minimum chunk length when adaptive splitting kicks in | `30` |
| `adaptiveNremove` | select harmonics via z-score instead of a fixed count | `True` |
| `fixedNremove` | minimum number of harmonics to remove | `1` |
| `noiseCompDetectSigma` | z-score threshold used by the adaptive remover | `3.0` |
| `maxHarmonics` | upper bound on harmonics evaluated per frequency | `5` |

## Development notes

- The harmonic regression design keeps the data rank intact: the noise estimate
  is projected out by subtracting the fitted sin/cos basis rather than
  filtering the data.
  
## License

MIT. See `LICENSE`.

