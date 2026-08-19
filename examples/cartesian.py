"""A 2-D (Cartesian) Fresnel propagator — for the examples, not for the core.

`diffractor`'s propagators are axisymmetric by construction: every one of them
is a Hankel/BOR operator on a radial grid.  A hexagon is not axisymmetric, so
this module supplies the missing piece the way `scattering/planar.py` says it
will eventually arrive ("Cartesian FFT version to follow the same pattern"):

    U(x′,y′) ∝ e^{ik|x′|²/2z} ∬ U₀(x,y) e^{ik|x|²/2z} e^{−2πi (x·x′)/(λz)} dx dy

evaluated by a *separable matrix DFT*, so the output window is chosen freely
instead of being handed to you by an FFT grid.  That matters here for two
reasons: every wavelength must land on the SAME physical screen for a colour
composite to mean anything, and the interesting part of a diffraction pattern
is usually a small window that an FFT would sample far too coarsely.

This is paraxial — it is the Fresnel integral, and `propagation.paraxial`'s gate
applies to it just as much (`fresnel_validity_distance` is the check).  It is
kept in `examples/` and out of the physical core because the core's contract is
that nothing enters it without a rung on the validation ladder; what this
module gets instead is `check_against_core()`, which puts it side by side with
the package's exact axisymmetric propagator on a circular aperture.
"""
import numpy as np

__all__ = ["regular_polygon_mask", "disc_mask", "fresnel_zoom",
           "check_against_core"]


def _supersampled(x, y, inside, supersample):
    """Area-coverage (grey-edge) evaluation of a hard mask."""
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    off = (np.arange(supersample) + 0.5) / supersample - 0.5
    acc = np.zeros((y.size, x.size))
    for oy in off:
        for ox in off:
            X, Y = np.meshgrid(x + ox * dx, y + oy * dy)
            acc += inside(X, Y)
    return acc / supersample**2


def regular_polygon_mask(x, y, n_sides, radius, *, orientation=np.pi / 2,
                         supersample=4):
    """Regular polygon of circumradius `radius`, as the intersection of n
    half-planes.  `orientation` is the angle of the first vertex."""
    normals = orientation + np.pi / n_sides + 2 * np.pi * np.arange(n_sides) / n_sides
    apothem = radius * np.cos(np.pi / n_sides)

    def inside(X, Y):
        keep = np.ones(X.shape, bool)
        for th in normals:
            keep &= (X * np.cos(th) + Y * np.sin(th)) <= apothem
        return keep

    return _supersampled(x, y, inside, supersample), normals


def disc_mask(x, y, radius, *, supersample=4):
    return _supersampled(x, y, lambda X, Y: X**2 + Y**2 <= radius**2, supersample)


def fresnel_zoom(U0, x, y, z, lam, x_out, y_out, n=1.0):
    """Fresnel integral onto a freely chosen output window (separable matrix DFT).

    Parameters
    ----------
    U0 : (ny, nx) complex
        Field in the aperture plane, on the tensor grid (y, x).
    x, y : 1-D arrays
        Input coordinates (uniform).  The field must be zero outside them —
        which is exact for an aperture in an opaque screen.
    z, lam, n :
        Propagation distance, vacuum wavelength, index of the medium.
    x_out, y_out : 1-D arrays
        Output coordinates.  Any window, any sampling.
    """
    k = 2 * np.pi * n / lam
    lz = (lam / n) * z
    Xi, Yi = np.meshgrid(x, y)
    Uq = U0 * np.exp(1j * k * (Xi**2 + Yi**2) / (2 * z))
    Ax = np.exp(-2j * np.pi * np.outer(x_out, x) / lz)        # (nx_out, nx)
    Ay = np.exp(-2j * np.pi * np.outer(y_out, y) / lz)        # (ny_out, ny)
    U = Ay @ Uq @ Ax.T                                        # (ny_out, nx_out)
    Xo, Yo = np.meshgrid(x_out, y_out)
    pre = np.exp(1j * k * z) / (1j * lz)
    return pre * np.exp(1j * k * (Xo**2 + Yo**2) / (2 * z)) * U * (x[1]-x[0]) * (y[1]-y[0])


def check_against_core(a=50.0, z=1.0e5, lam=0.55, n_in=768, n_out=241,
                       half_width=None):
    """Same circular aperture, two ways: this 2-D Fresnel integral against the
    package's exact (non-paraxial) Rayleigh–Sommerfeld propagator.

    Returns (r, I_2d, I_core, max relative difference of the profiles).
    """
    from diffractor.propagation import rs1_plane

    half_width = half_width if half_width is not None else 8.0 * lam * z / (2 * a)
    x = np.linspace(-1.05 * a, 1.05 * a, n_in)
    U0 = disc_mask(x, x, a).astype(complex)
    xo = np.linspace(-half_width, half_width, n_out)
    U = fresnel_zoom(U0, x, x, z, lam, xo, np.zeros(1))
    I2d = np.abs(U[0]) ** 2

    r = np.linspace(0.0, a, 3001)
    Ur = np.ones_like(r, dtype=complex)
    I_core = np.abs(rs1_plane(Ur, r, z, 1.0, lam, np.abs(xo))) ** 2

    scale = I_core.max()
    return xo, I2d / scale, I_core / scale, np.abs(I2d - I_core).max() / scale


if __name__ == "__main__":
    xo, i2, ic, err = check_against_core()
    print(f"2-D Fresnel vs the core's exact Rayleigh–Sommerfeld: "
          f"max difference {err:.2e} of the peak")
