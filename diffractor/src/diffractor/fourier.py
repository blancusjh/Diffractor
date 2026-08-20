"""The 2-D Fourier transform, as an operator on fields.

One transform, several representations.  With angular wavenumber k and the
optics sign convention,

    FT2[f](k)  = ∬ f(x) e^{−i k·x} d²x
    IFT2[F](x) = (2π)^{−2} ∬ F(k) e^{+i k·x} d²k,

and how the double integral is *evaluated* is decided by the basis of the
field's grid — the transform itself never changes:

* **cartesian** — separable.  On the grid's own reciprocal (``kgrid=None``)
  the Riemann sum over uniform axes is exactly a centred FFT scaled by dx·dy;
  on an explicit cartesian ``kgrid`` it is a separable matrix DFT, which is
  what lets an output window be *chosen* (the zoom that resolves a focal
  spot) instead of dictated by the FFT.
* **polar** — the angular decomposition.  By Jacobi–Anger the m-th angular
  Fourier mode transforms through the m-th order Hankel transform:

      f(r,θ) = Σ_m f_m(r) e^{imθ},   f_m = (1/2π) ∫ f e^{−imθ} dθ
      FT2[f](k,φ) = Σ_m [ 2π (−i)^m H_m{f_m}(k) ] e^{imφ}
      IFT2 mirrors with (i^m / 2π) H_m^{-1}.

  Since (−i)^{−m} J_{−m} = (−i)^m J_m, the ±m modes share one kernel *and*
  one phase factor: only |m| matters, and the even-n_θ Nyquist mode — an
  unknowable mixture of ±n_θ/2 — is transformed exactly regardless of the
  split.  Axisymmetry (n_θ = 1) degenerates to the plain order-zero Hankel
  transform with no special casing.
* **custom / cross-basis** — the transform written as what it is: a weighted
  sum over the sample nodes at the requested frequency nodes, both addressed
  through their cartesian images.  Exact, O(N·M), and the fallback that makes
  any (grid, kgrid) pairing legal.

The polar kernels J_|m|(k⊗r) depend only on the (r, k) axes — never on the
wavelength — so a :class:`PolarPlan` computed once serves the forward and the
inverse transform (transposed) of every spectral line.  Propagators, which
transform back and forth across the same pair, build one plan and pay the
Bessel evaluations once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .basis import CARTESIAN, POLAR
from .field import Field
from .hankel import hankel_matrix, hankel_transform, inverse_hankel_transform
from .space import Grid

__all__ = ["FT2", "IFT2", "PolarPlan", "PLAN_CACHE_BYTES"]

#: A PolarPlan bigger than this refuses to materialise its kernel matrices and
#: the transform falls back to chunked evaluation — a memory guard, not a
#: performance knob.
PLAN_CACHE_BYTES: int = 1 << 30


# ══════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════
def _is_uniform(axis: np.ndarray) -> bool:
    if axis.size < 2:
        return False
    d = np.diff(axis)
    return bool(np.allclose(d, d[0], rtol=1e-9, atol=0.0))


def _axis_weights(axis: np.ndarray) -> np.ndarray:
    """Quadrature weights of one axis: uniform → Riemann (matches the FFT sum
    exactly, which keeps the FFT and matrix paths identical to machine
    precision); non-uniform → composite trapezoid."""
    if axis.size == 1:
        return np.ones(1)
    if _is_uniform(axis):
        return np.full(axis.size, float(axis[1] - axis[0]))
    w = np.empty(axis.size)
    w[0] = (axis[1] - axis[0]) / 2.0
    w[-1] = (axis[-1] - axis[-2]) / 2.0
    w[1:-1] = (axis[2:] - axis[:-2]) / 2.0
    return w


def _require_domain(field: Field, needed: str, op: str) -> None:
    if field.domain != needed:
        raise ValueError(
            f"{op} transforms a {needed!r}-domain field; got {field.domain!r} "
            f"(the operators flip the domain — use the other one)")


def _angular_modes(theta: np.ndarray) -> np.ndarray:
    """Signed mode numbers in FFT order for the uniform circle sampling."""
    n = theta.size
    return np.rint(np.fft.fftfreq(n) * n).astype(int)


# ══════════════════════════════════════════════════════════════════════════
#  Polar plan
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, eq=False)
class PolarPlan:
    """Cached Hankel kernels for one (polar space grid, polar k-grid) pair.

    Holds one ``J_|m|(k ⊗ r)`` matrix per distinct ``|m| = 0 … n_θ//2``.  The
    same matrices serve FT2 (as built) and IFT2 (transposed), and every
    spectral line — they depend only on the radial axes.  Build one when the
    same transform pair is applied repeatedly (a propagator, a z-sweep).
    """

    space_grid: Grid
    kgrid: Grid
    matrices: Tuple[np.ndarray, ...]

    @classmethod
    def build(cls, space_grid: Grid, kgrid: Grid, *,
              max_bytes: int = PLAN_CACHE_BYTES) -> Optional["PolarPlan"]:
        """The plan for the pair, or ``None`` if it would exceed ``max_bytes``
        (callers then fall back to chunked evaluation)."""
        _check_polar_pair(space_grid, kgrid)
        r, k = space_grid.axes[0], kgrid.axes[0]
        n_orders = space_grid.axes[1].size // 2 + 1
        if n_orders * r.size * k.size * 8 > max_bytes:
            return None
        mats = tuple(hankel_matrix(a, r, k) for a in range(n_orders))
        return cls(space_grid, kgrid, mats)


def _check_polar_pair(space_grid: Grid, kgrid: Grid) -> None:
    if kgrid.basis is not POLAR:
        raise ValueError("the polar transform's kgrid must be polar")
    if (kgrid.axes[1].size != space_grid.axes[1].size
            or not np.array_equal(kgrid.axes[1], space_grid.axes[1])):
        raise ValueError(
            "a polar kgrid must reuse the spatial grid's theta axis verbatim "
            "— the angular modes of f and of its transform are the same modes")


def _plan_matrix(plan: Optional[PolarPlan], grid: Grid, kgrid: Grid,
                 order: int) -> Optional[np.ndarray]:
    if plan is None:
        return None
    if plan.space_grid is not grid and not (
            np.array_equal(plan.space_grid.axes[0], grid.axes[0])
            and np.array_equal(plan.space_grid.axes[1], grid.axes[1])):
        raise ValueError("plan was built for a different spatial grid")
    if plan.kgrid is not kgrid and not np.array_equal(plan.kgrid.axes[0],
                                                      kgrid.axes[0]):
        raise ValueError("plan was built for a different kgrid")
    return plan.matrices[order]


# ══════════════════════════════════════════════════════════════════════════
#  Polar path (Jacobi–Anger + Hankel)
# ══════════════════════════════════════════════════════════════════════════
def _polar_transform(field: Field, kgrid: Grid, plan: Optional[PolarPlan],
                     inverse: bool) -> np.ndarray:
    """Both directions of the polar transform; ``kgrid`` is the target grid
    (a k-grid forward, a space grid inverse)."""
    src_grid = field.grid
    n_theta = src_grid.axes[1].size
    n_out = kgrid.axes[0].size
    values = field.values                                # (n_r, n_θ, n_λ)
    n_lambda = values.shape[-1]

    modes = np.fft.fft(values, axis=1) / n_theta          # f_m in FFT order
    m_of = _angular_modes(src_grid.axes[1])
    out_modes = np.empty((n_out, n_theta, n_lambda), complex)

    for a in range(n_theta // 2 + 1):
        cols = np.nonzero(np.abs(m_of) == a)[0]
        if cols.size == 0:
            continue
        block = modes[:, cols, :].reshape(modes.shape[0], -1)
        if inverse:
            # plan matrices are forward (n_k, n_r); grids arrive swapped here
            matrix = _plan_matrix(plan, kgrid, src_grid, a)
            out = inverse_hankel_transform(block, src_grid.axes[0],
                                           kgrid.axes[0], order=a,
                                           matrix=matrix)
            factor = (1j) ** a / (2.0 * np.pi)
        else:
            matrix = _plan_matrix(plan, src_grid, kgrid, a)
            out = hankel_transform(block, src_grid.axes[0], kgrid.axes[0],
                                   order=a, matrix=matrix)
            factor = 2.0 * np.pi * (-1j) ** a
        out_modes[:, cols, :] = factor * out.reshape(n_out, cols.size, n_lambda)

    return n_theta * np.fft.ifft(out_modes, axis=1)


# ══════════════════════════════════════════════════════════════════════════
#  Cartesian paths
# ══════════════════════════════════════════════════════════════════════════
def _cartesian_fft(field: Field, inverse: bool) -> Tuple[np.ndarray, Grid]:
    """Centred FFT on the grid's own reciprocal (exact Riemann sum).

    Works for any *uniform* axes, centred or not: the axis offset x₀ becomes
    the linear phase e^{∓i k x₀} — bookkeeping, not approximation.
    """
    grid = field.grid
    if inverse:
        kx, ky = grid.axes
        space = grid.reciprocal_source  # set by FT2 below?  no — see IFT2 note
    x_axis, y_axis = grid.axes
    dx = x_axis[1] - x_axis[0]
    dy = y_axis[1] - y_axis[0]
    values = field.values

    if not inverse:
        target = grid.reciprocal()
        kx, ky = target.axes
        G = np.fft.fft2(values, axes=(0, 1))
        G = np.fft.fftshift(G, axes=(0, 1))
        phase = (np.exp(-1j * kx * x_axis[0])[:, None, None]
                 * np.exp(-1j * ky * y_axis[0])[None, :, None])
        return G * phase * (dx * dy), target
    raise AssertionError("inverse handled by _cartesian_ifft")


def _cartesian_ifft(field: Field, space_grid: Grid) -> np.ndarray:
    """Centred inverse FFT back onto ``space_grid`` (the grid whose reciprocal
    the spectrum lives on)."""
    kx, ky = field.grid.axes
    x_axis, y_axis = space_grid.axes
    dx = x_axis[1] - x_axis[0]
    dy = y_axis[1] - y_axis[0]
    phase = (np.exp(+1j * kx * x_axis[0])[:, None, None]
             * np.exp(+1j * ky * y_axis[0])[None, :, None])
    G = np.fft.ifftshift(field.values * phase, axes=(0, 1))
    return np.fft.ifft2(G, axes=(0, 1)) / (dx * dy)


def _separable_dft(field: Field, kgrid: Grid, inverse: bool) -> np.ndarray:
    """Separable matrix DFT between two cartesian grids (any axes).

    Two GEMMs with the spectral axis folded in — the same arithmetic as the
    obvious einsum, but routed through BLAS."""
    src = field.grid
    sign = +1j if inverse else -1j
    wx = _axis_weights(src.axes[0])
    wy = _axis_weights(src.axes[1])
    Ex = np.exp(sign * np.outer(kgrid.axes[0], src.axes[0])) * wx  # (Mx, Nx)
    Ey = np.exp(sign * np.outer(kgrid.axes[1], src.axes[1])) * wy  # (My, Ny)
    vals = field.values                                            # (Nx, Ny, L)
    nx, ny, nl = vals.shape
    mx, my = Ex.shape[0], Ey.shape[0]
    out = (Ex @ vals.reshape(nx, ny * nl)).reshape(mx, ny, nl)     # (Mx, Ny, L)
    out = (Ey @ out.transpose(1, 0, 2).reshape(ny, mx * nl))       # (My, Mx*L)
    out = out.reshape(my, mx, nl).transpose(1, 0, 2)               # (Mx, My, L)
    if inverse:
        out /= (2.0 * np.pi) ** 2
    return np.ascontiguousarray(out)


def _generic_dft(field: Field, kgrid: Grid, inverse: bool,
                 weights: Optional[np.ndarray], chunk: int = 2048) -> np.ndarray:
    """The transform as a weighted sum over nodes, through cartesian images.

    Exact for any (grid, kgrid) basis pairing; O(N·M) memory-chunked.  The
    source grid must provide an area measure — its own ``weights()`` or the
    explicit ``weights`` argument (required for custom bases).
    """
    src = field.grid
    if weights is None:
        weights = src.weights()
    X, Y = src.cartesian_meshes()
    KX, KY = kgrid.cartesian_meshes()
    sign = +1j if inverse else -1j
    xf, yf = X.ravel(), Y.ravel()
    kxf, kyf = KX.ravel(), KY.ravel()
    vals = (field.values * weights[..., None]).reshape(xf.size, -1)
    out = np.empty((kxf.size, vals.shape[1]), complex)
    for i in range(0, kxf.size, chunk):
        E = np.exp(sign * (np.outer(kxf[i:i + chunk], xf)
                           + np.outer(kyf[i:i + chunk], yf)))
        out[i:i + chunk] = E @ vals
    out = out.reshape((*kgrid.shape, field.values.shape[-1]))
    if inverse:
        out /= (2.0 * np.pi) ** 2
    return out


# ══════════════════════════════════════════════════════════════════════════
#  The operators
# ══════════════════════════════════════════════════════════════════════════
def FT2(field: Field, *, kgrid: Optional[Grid] = None,
        plan: Optional[PolarPlan] = None,
        weights: Optional[np.ndarray] = None) -> Field:
    """Forward transform of a space-domain field onto a frequency grid.

    ``kgrid=None`` transforms onto ``field.grid.reciprocal()`` by the fast
    path of the field's basis (FFT / per-mode Hankel).  An explicit ``kgrid``
    evaluates the same integral on the frequencies *you* chose — matrix DFT
    for separable pairings, the generic node sum otherwise.  ``weights``
    supplies the area measure for custom-basis grids; ``plan`` reuses polar
    kernels across repeated transforms.
    """
    _require_domain(field, "space", "FT2")
    grid = field.grid

    if grid.basis is CARTESIAN:
        if kgrid is None:
            values, target = _cartesian_fft(field, inverse=False)
            return field.like(values, grid=target, domain="frequency")
        if kgrid.basis is CARTESIAN:
            return field.like(_separable_dft(field, kgrid, inverse=False),
                              grid=kgrid, domain="frequency")
        return field.like(_generic_dft(field, kgrid, False, weights),
                          grid=kgrid, domain="frequency")

    if grid.basis is POLAR:
        if kgrid is None:
            kgrid = grid.reciprocal()
        if kgrid.basis is POLAR:
            _check_polar_pair(grid, kgrid)
            values = _polar_transform(field, kgrid, plan, inverse=False)
            return field.like(values, grid=kgrid, domain="frequency")
        return field.like(_generic_dft(field, kgrid, False, weights),
                          grid=kgrid, domain="frequency")

    if kgrid is None:
        raise ValueError(
            f"a {grid.basis.name!r}-basis grid has no default reciprocal; "
            f"pass an explicit kgrid (and weights)")
    return field.like(_generic_dft(field, kgrid, False, weights),
                      grid=kgrid, domain="frequency")


def IFT2(field: Field, *, grid: Optional[Grid] = None,
         plan: Optional[PolarPlan] = None,
         weights: Optional[np.ndarray] = None) -> Field:
    """Inverse transform of a frequency-domain field onto a space grid.

    ``grid`` is the target space grid; it is required, because a frequency
    grid does not remember where its samples came from — the caller does.
    Fast paths mirror :func:`FT2`: FFT when the k-grid is the target's own
    reciprocal, per-mode Hankel for polar pairs, matrix/node sums otherwise.
    """
    _require_domain(field, "frequency", "IFT2")
    if grid is None:
        raise ValueError("IFT2 needs the target space grid: IFT2(F, grid=...)")
    kgrid = field.grid

    if kgrid.basis is CARTESIAN and grid.basis is CARTESIAN:
        native = grid.reciprocal() if all(map(_is_uniform, grid.axes)) else None
        if (native is not None
                and native.shape == kgrid.shape
                and np.allclose(native.axes[0], kgrid.axes[0])
                and np.allclose(native.axes[1], kgrid.axes[1])):
            return field.like(_cartesian_ifft(field, grid), grid=grid,
                              domain="space")
        return field.like(_separable_dft(field, grid, inverse=True),
                          grid=grid, domain="space")

    if kgrid.basis is POLAR and grid.basis is POLAR:
        _check_polar_pair(grid, kgrid)
        values = _polar_transform(field, grid, plan, inverse=True)
        return field.like(values, grid=grid, domain="space")

    return field.like(_generic_dft(field, grid, True, weights),
                      grid=grid, domain="space")
