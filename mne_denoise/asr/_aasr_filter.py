"""Yule-Walker statistics-filter design for Adaptive ASR.

These helpers reproduce the MATLAB AASR ``datafiltering2`` /
``eegfiltfft(1, 50)`` pre-emphasis as an 8th-order Yule-Walker IIR filter. They
are factored out of :mod:`mne_denoise.asr.adaptive` so the estimator module
stays focused on the adaptive learning logic; only :func:`design_aasr_filter`
is used downstream.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.linalg import toeplitz


def _polystab(a: np.ndarray) -> np.ndarray:
    """Stabilize a polynomial by reflecting roots inside the unit circle."""
    roots = np.roots(a)
    keep = roots != 0
    reflect = 0.5 * (np.sign(np.abs(roots[keep]) - 1.0) + 1.0)
    roots[keep] = (1.0 - reflect) * roots[keep] + reflect / np.conj(roots[keep])
    nz = np.flatnonzero(a != 0)
    b = a[nz[0]] * np.poly(roots)
    if not np.any(np.imag(a)):
        b = np.real(b)
    return np.asarray(b, dtype=np.float64)


def _numf(h: np.ndarray, a: np.ndarray, nb: int) -> np.ndarray:
    """Least-squares FIR numerator matching impulse response ``h`` given ``a``."""
    nh = int(np.max(h.size))
    impulse = np.zeros(nh, dtype=np.float64)
    impulse[0] = 1.0
    impr = signal.lfilter(np.array([1.0]), a, impulse)
    rhs = np.concatenate(([1.0], np.zeros(nb, dtype=np.float64)))
    return np.linalg.lstsq(toeplitz(impr, rhs), h.T, rcond=None)[0].T


def _denf(R: np.ndarray, na: int) -> np.ndarray:
    """Least-squares AR denominator from autocorrelation ``R``."""
    nr = int(np.max(np.size(R)))
    Rm = toeplitz(R[na : nr - 1], R[na:0:-1])
    rhs = -R[na + 1 : nr]
    return np.concatenate(([1.0], np.linalg.lstsq(Rm, rhs.T, rcond=None)[0].T))


def _yulewalk(
    order: int, F: np.ndarray, M: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Yule-Walker IIR design from a piecewise-linear magnitude template."""
    F = np.asarray(F, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    npt = 513
    lap = int(np.fix((npt - 1) / 25))
    Ht = np.zeros(npt, dtype=np.float64)
    Ht[0] = M[0]
    df = np.diff(F)

    nb = 0
    for idx in range(F.size - 1):
        if df[idx] == 0:
            nb = nb - int(lap / 2)
            ne = nb + lap
        else:
            ne = int(np.fix(F[idx + 1] * npt)) - 1
        j = np.arange(nb, ne + 1)
        inc = 0.0 if ne == nb else (j - nb) / (ne - nb)
        Ht[nb : ne + 1] = inc * M[idx + 1] + (1.0 - inc) * M[idx]
        nb = ne + 1

    Ht = np.concatenate([Ht, Ht[-2:0:-1]])
    n = Ht.size
    n2 = int(np.fix((n + 1) / 2))
    nr = 4 * order
    nt = np.arange(nr, dtype=np.float64)

    R = np.real(np.fft.ifft(Ht * Ht))
    R = R[:nr] * (0.54 + 0.46 * np.cos(np.pi * nt / max(nr - 1, 1)))

    Rwindow = np.concatenate(
        (
            np.array([0.5], dtype=np.float64),
            np.ones(max(n2 - 1, 0), dtype=np.float64),
            np.zeros(max(n - n2, 0), dtype=np.float64),
        )
    )
    A = _polystab(_denf(R, order))
    Qh = _numf(np.concatenate(([R[0] / 2.0], R[1:nr])), A, order)

    _, Ss = signal.freqz(Qh, A, worN=n, whole=True)
    Ss = np.maximum(2.0 * np.real(Ss), np.finfo(float).eps)
    hh = np.fft.ifft(np.exp(np.fft.fft(Rwindow * np.fft.ifft(np.log(Ss)))))
    B = np.real(_numf(hh[:nr], A, order))
    return np.asarray(B, dtype=np.float64), np.asarray(A, dtype=np.float64)


def design_aasr_filter(sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Design the AASR 1-50 Hz pre-emphasis statistics filter (b, a)."""
    freqs = (
        np.array(
            [
                0.0,
                2.0,
                3.0,
                13.0,
                16.0,
                40.0,
                min(80.0, (sfreq / 2.0) - 1.0),
                sfreq / 2.0,
            ],
            dtype=np.float64,
        )
        * 2.0
        / sfreq
    )
    mags = np.array([3.0, 0.75, 0.33, 0.33, 1.0, 1.0, 3.0, 3.0], dtype=np.float64)
    return _yulewalk(8, freqs, mags)
