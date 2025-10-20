Getting started
===============

Installation
------------

Install the latest release from PyPI:

.. code-block:: console

   pip install mne-denoise

For development work, clone the repository and install the optional extras:

.. code-block:: console

   pip install -e .[dev,mne,docs]
   pre-commit install

Basic usage
-----------

.. code-block:: python

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

MNE integration
---------------

.. code-block:: python

   import mne
   from mne_denoise import apply_zapline_to_raw

   raw = mne.io.read_raw_fif("mne_sample.fif", preload=True)
   raw_clean, *_ = apply_zapline_to_raw(raw, line_freqs="line", plotResults=False)
   raw_clean.plot()

Example gallery
---------------

The repository ships with runnable scripts in ``examples/``. Start with
``examples/basic_usage.py`` to see the functional API end-to-end. Future
releases will extend this into a full gallery rendered with the documentation.
