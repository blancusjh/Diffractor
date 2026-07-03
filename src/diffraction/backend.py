"""Array backend selection for CPU (NumPy) or GPU (CuPy) execution.

The numerical core is written against the NumPy API. CuPy exposes the same
API on NVIDIA GPUs, so backend-aware code paths run unchanged on either device
by resolving the *array module* (``numpy`` or ``cupy``) from the arrays they
are given. This module centralizes that resolution and the host/device
transfers. The GPU-capable entry points are :class:`~diffraction.asm.AngularSpectrum`
(the batched propagation engine, built for device execution) and the array
operators ``FFT2`` / ``IFFT2``; the one-shot functional propagators are
NumPy-first.

If CuPy is not installed the toolkit still works: every helper degrades to a
NumPy-only path and :data:`CUPY_AVAILABLE` is ``False``.
"""

from __future__ import annotations

import numpy as np

try:  # CuPy is an optional dependency (see the ``gpu`` extra).
    import cupy as _cp

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without a GPU stack
    _cp = None
    CUPY_AVAILABLE = False

__all__ = [
    "CUPY_AVAILABLE",
    "array_module",
    "asnumpy",
    "resolve_module",
    "to_device",
]


def array_module(*arrays):
    """Return the array module (``numpy`` or ``cupy``) that owns ``arrays``.

    Mirrors :func:`cupy.get_array_module`: if every argument is a NumPy array
    (or CuPy is absent) this returns ``numpy``; if any argument lives on the
    GPU it returns ``cupy``.
    """
    if CUPY_AVAILABLE:
        return _cp.get_array_module(*arrays)
    return np


def resolve_module(device: str):
    """Return the array module for a named ``device`` (``"cpu"`` or ``"gpu"``)."""
    device = device.lower()
    if device in ("cpu", "numpy", "np"):
        return np
    if device in ("gpu", "cuda", "cupy"):
        if not CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is not available; install the 'gpu' extra "
                "(e.g. pip install cupy-cuda12x) to run on the GPU."
            )
        return _cp
    raise ValueError(f"Unknown device {device!r}; use 'cpu' or 'gpu'.")


def to_device(a, device: str):
    """Move an array to ``"cpu"`` (NumPy) or ``"gpu"`` (CuPy)."""
    xp = resolve_module(device)
    if xp is np:
        return asnumpy(a)
    return xp.asarray(a)


def asnumpy(a) -> np.ndarray:
    """Return a NumPy array from ``a``, copying off the GPU if necessary."""
    if CUPY_AVAILABLE:
        return _cp.asnumpy(a)
    return np.asarray(a)
