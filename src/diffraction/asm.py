"""Reusable, batched angular-spectrum propagator.

:func:`diffraction.propagation.asm_propagator` is a stateless one-shot call:
it rebuilds the transfer-function machinery on every invocation. When the same
field is propagated to *many* distances — a focal z-sweep, a movie, a depth
stack — that rebuild dominates the cost.

:class:`AngularSpectrum` precomputes everything that does not depend on ``z``
(the spatial-frequency grids, the longitudinal wavenumber ``kz`` and the FFT of
the input field) once, so each additional plane costs a single inverse FFT. The
same object runs on the CPU (NumPy) or, if the input field is a CuPy array or
``device="gpu"`` is requested, on an NVIDIA GPU — where the batched transform
is typically one to two orders of magnitude faster.

The physics matches :func:`asm_propagator` exactly: the exact scalar transfer
function ``exp(i kz z)``, the Matsushima–Shimobaba band limit and optional
evanescent-wave decay. A regression test pins the two implementations together.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .backend import array_module, asnumpy, resolve_module
from .grids import Array, Grid, grid_spacing

__all__ = ["AngularSpectrum"]


def _fft2c(xp, g):
    """Centered 2D FFT on the array module ``xp``."""
    return xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(g)))


def _ifft2c(xp, G):
    """Centered inverse 2D FFT on the array module ``xp``."""
    return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(G)))


class AngularSpectrum:
    """Precomputed angular-spectrum propagator for repeated / batched use.

    Parameters
    ----------
    U : 2D complex array
        Input field sampled on ``grid`` (NumPy or CuPy).
    grid : (x, y)
        Spatial coordinate grids from :func:`numpy.meshgrid`.
    wavelength : float
        Vacuum wavelength [m].
    n : float
        Refractive index of the propagation medium.
    include_evanescent : bool
        Keep evanescent components (exponential decay) instead of filtering
        them out. Default ``False``.
    bandlimit : bool
        Apply the Matsushima–Shimobaba band limit (default ``True``); zeroes
        the frequencies whose sampled transfer-function phase aliases at the
        requested distance.
    pad_factor : int
        Zero-pad the field by this factor before propagating and crop
        afterwards (default 1), suppressing circular-convolution wrap-around.
    device : {"cpu", "gpu"} or None
        Force a backend. ``None`` (default) keeps the field on whatever device
        it already lives on: a NumPy field stays on the CPU, a CuPy field on
        the GPU.

    Notes
    -----
    The input FFT is computed once at construction, so mutating ``U`` after
    building the propagator has no effect — build a new one instead.
    """

    def __init__(
        self,
        U: Array,
        grid: Grid,
        wavelength: float,
        n: float = 1.0,
        *,
        include_evanescent: bool = False,
        bandlimit: bool = True,
        pad_factor: int = 1,
        device: Optional[str] = None,
    ) -> None:
        if wavelength <= 0:
            raise ValueError("wavelength must be positive.")
        if n <= 0:
            raise ValueError("n must be positive.")
        if pad_factor < 1:
            raise ValueError("pad_factor must be at least 1.")

        xp = resolve_module(device) if device is not None else array_module(U)
        self.xp = xp
        self.include_evanescent = include_evanescent
        self.bandlimit = bandlimit
        self.pad_factor = pad_factor
        self.wavelength_medium = wavelength / n

        dx, dy = grid_spacing(grid)
        self._dx, self._dy = dx, dy

        U = xp.asarray(U, dtype=xp.complex128)
        self._ny, self._nx = U.shape

        if pad_factor > 1:
            self._pad_x = ((pad_factor - 1) * self._nx) // 2
            self._pad_y = ((pad_factor - 1) * self._ny) // 2
            U = xp.pad(
                U,
                ((self._pad_y, self._pad_y), (self._pad_x, self._pad_x)),
            )
        else:
            self._pad_x = self._pad_y = 0

        NY, NX = U.shape
        self._NY, self._NX = NY, NX

        fx1d = xp.fft.fftshift(xp.fft.fftfreq(NX, d=dx))
        fy1d = xp.fft.fftshift(xp.fft.fftfreq(NY, d=dy))
        fx, fy = xp.meshgrid(fx1d, fy1d)
        self._fx, self._fy = fx, fy

        f2 = fx**2 + fy**2
        cutoff = (1.0 / self.wavelength_medium) ** 2

        if include_evanescent:
            self._kz = 2.0 * np.pi * xp.sqrt(cutoff - f2 + 0.0j)
            self._propagating = None
        else:
            self._propagating = f2 <= cutoff
            self._kz = 2.0 * np.pi * xp.sqrt(xp.maximum(cutoff - f2, 0.0))

        self._U_fft = _fft2c(xp, U)

    def _transfer(self, z: float):
        xp = self.xp
        if self.include_evanescent:
            H = xp.exp(1.0j * self._kz * z)
        else:
            H = xp.where(self._propagating, xp.exp(1.0j * self._kz * z), 0.0)

        if self.bandlimit and z != 0:
            dfx = 1.0 / (self._NX * self._dx)
            dfy = 1.0 / (self._NY * self._dy)
            lam = self.wavelength_medium
            fx_limit = 1.0 / (lam * np.sqrt((2.0 * dfx * abs(z)) ** 2 + 1.0))
            fy_limit = 1.0 / (lam * np.sqrt((2.0 * dfy * abs(z)) ** 2 + 1.0))
            H = xp.where(
                (xp.abs(self._fx) <= fx_limit) & (xp.abs(self._fy) <= fy_limit),
                H,
                0.0,
            )
        return H

    def _crop(self, Uz):
        if self.pad_factor > 1:
            return Uz[
                self._pad_y : self._pad_y + self._ny,
                self._pad_x : self._pad_x + self._nx,
            ]
        return Uz

    def propagate(self, z: float) -> Array:
        """Field at distance ``z``, on the propagator's device.

        Negative ``z`` back-propagates. The result shares the array module of
        the input field (NumPy on the CPU, CuPy on the GPU).
        """
        Uz = _ifft2c(self.xp, self._U_fft * self._transfer(z))
        return self._crop(Uz)

    def propagate_stack(self, zs: Sequence[float]) -> List[Array]:
        """Fields at every distance in ``zs`` (list of on-device arrays)."""
        return [self.propagate(float(z)) for z in zs]

    def intensity_stack(
        self,
        zs: Sequence[float],
        *,
        normalize: bool = True,
        to_cpu: bool = True,
    ) -> List[Array]:
        """Intensity frames ``|U(z)|²`` for every ``z`` in ``zs``.

        Convenience for building z-sweep animations: returns real-valued
        frames, individually peak-normalized by default and moved back to the
        host (``to_cpu=True``) so they can be handed straight to
        :func:`diffraction.viz.animate` or matplotlib.
        """
        xp = self.xp
        frames = []
        for z in zs:
            I = xp.abs(self.propagate(float(z))) ** 2
            if normalize:
                peak = I.max()
                if peak > 0:
                    I = I / peak
            frames.append(asnumpy(I) if to_cpu else I)
        return frames
