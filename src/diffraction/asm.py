"""Reusable, batched angular-spectrum propagator.

:func:`diffraction.propagation.asm_propagator` is a stateless one-shot call:
it rebuilds the transfer-function machinery on every invocation. When the same
field is propagated to *many* distances — a focal z-sweep, a movie, a depth
stack — that rebuild dominates the cost.

:class:`AngularSpectrum` precomputes everything that does not depend on ``z``
(the spatial-frequency grid, the longitudinal wavenumber ``kz`` and the FFT of
the input field) once, so each additional plane costs a single inverse FFT. The
same object runs on the CPU (NumPy) or, if the input field is a CuPy array or
``device="gpu"`` is requested, on an NVIDIA GPU.

With an explicit ``output_grid`` the inverse transform is sampled on a chosen
grid (via a precomputed matrix DFT), decoupling the output window/sampling from
the input — the MPASM / scaled-ASM capability.

The physics matches :func:`asm_propagator` exactly on the native grid.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .backend import array_module, asnumpy, resolve_module
from .field import Field
from .grids import Array, Grid

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
    field : Field
        Input field (NumPy or CuPy values).
    wavelength : float
        Vacuum wavelength [m].
    n : float
        Refractive index of the propagation medium.
    output_grid : Grid, optional
        If given, every plane is sampled on this grid via a precomputed matrix
        DFT (decoupled output sampling; MPASM). ``None`` (default) returns each
        plane on the input grid using the fast inverse FFT.
    include_evanescent : bool
        Keep evanescent components (exponential decay). Default ``False``.
    bandlimit : bool
        Apply the Matsushima–Shimobaba band limit (default ``True``).
    pad_factor : int
        Zero-pad the field by this factor before propagating (default 1),
        suppressing circular-convolution wrap-around.
    device : {"cpu", "gpu"} or None
        Force a backend. ``None`` keeps the field on its current device.

    Notes
    -----
    The input FFT is computed once at construction, so mutating the field
    afterwards has no effect — build a new propagator instead.
    """

    def __init__(
        self,
        field: Field,
        wavelength: float,
        n: float = 1.0,
        *,
        output_grid: Optional[Grid] = None,
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

        grid = field.grid
        U = field.values
        xp = resolve_module(device) if device is not None else array_module(U)
        self.xp = xp
        self._grid = grid
        self.include_evanescent = include_evanescent
        self.bandlimit = bandlimit
        self.pad_factor = pad_factor
        self.output_grid = output_grid
        self.wavelength_medium = wavelength / n

        dx, dy = grid.spacing
        self._dx, self._dy = dx, dy

        U = xp.asarray(U, dtype=xp.complex128)
        self._ny, self._nx = U.shape

        if pad_factor > 1:
            self._pad_x = ((pad_factor - 1) * self._nx) // 2
            self._pad_y = ((pad_factor - 1) * self._ny) // 2
            U = xp.pad(U, ((self._pad_y, self._pad_y), (self._pad_x, self._pad_x)))
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

        # Precompute the inverse matrix-DFT operators for a decoupled output.
        if output_grid is not None:
            src_fx = fx1d  # padded native frequency axes
            src_fy = fy1d
            dst_x = xp.asarray(output_grid.x[0, :])
            dst_y = xp.asarray(output_grid.y[:, 0])
            self._Ex_inv = xp.exp(2.0j * np.pi * xp.outer(dst_x, src_fx))  # (Mx, NX)
            self._Ey_inv = xp.exp(2.0j * np.pi * xp.outer(dst_y, src_fy))  # (My, NY)
            self._inv_norm = 1.0 / (NY * NX)

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

    def propagate(self, z: float) -> Field:
        """Field at distance ``z``.

        Negative ``z`` back-propagates. The result carries the input grid (or
        ``output_grid`` if one was set) and shares the input's array module.
        """
        spectrum = self._U_fft * self._transfer(z)
        if self.output_grid is None:
            Uz = self._crop(_ifft2c(self.xp, spectrum))
            return Field(self._grid, Uz)
        Uz = (self._Ey_inv @ spectrum @ self._Ex_inv.T) * self._inv_norm
        return Field(self.output_grid, Uz)

    def propagate_stack(self, zs: Sequence[float]) -> List[Field]:
        """Fields at every distance in ``zs``."""
        return [self.propagate(float(z)) for z in zs]

    def intensity_stack(
        self,
        zs: Sequence[float],
        *,
        normalize: bool | str = True,
        to_cpu: bool = True,
    ) -> List[Array]:
        """Intensity frames ``|U(z)|²`` for every ``z`` in ``zs``.

        Convenience for building z-sweep animations: returns real-valued
        frames moved back to the host (``to_cpu=True``) so they can be handed
        straight to :func:`diffraction.viz.animate` or matplotlib.

        ``normalize`` selects the scaling: ``True`` (default) peak-normalizes
        each frame *individually* — every plane uses the full display range,
        but relative brightness between planes is lost; ``"global"`` divides
        the whole stack by its single maximum, preserving how a focus
        brightens through the sweep (matching
        :func:`~diffraction.longitudinal.longitudinal_field`); ``False``
        returns raw ``|U|²``.
        """
        if normalize not in (True, False, "global"):
            raise ValueError("normalize must be True, False or 'global'.")
        xp = self.xp
        frames = []
        for z in zs:
            I = xp.abs(self.propagate(float(z)).values) ** 2
            if normalize is True:
                peak = I.max()
                if peak > 0:
                    I = I / peak
            frames.append(asnumpy(I) if to_cpu else I)
        if normalize == "global":
            peak = max(float(f.max()) for f in frames)
            if peak > 0:
                frames = [f / peak for f in frames]
        return frames
