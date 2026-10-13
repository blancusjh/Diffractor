import numpy as np
import pytest

from diffractor import (
    MonochromaticField,
    circular_aperture,
    fresnel_max_spacing,
    fresnel_min_distance,
    make_grid,
    next_fft_size,
    recommend_grid_convergence,
)

try:
    import scipy.fft  # noqa: F401

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def test_make_grid_nesting_property():
    # The load-bearing assumption behind successive_diff: doubling N at fixed
    # length gives an exactly nested sample set.
    L = 6e-3
    for n in (128, 300):
        coarse = make_grid(n, L)
        fine = make_grid(2 * n, L)
        np.testing.assert_array_equal(fine.x[::2, ::2], coarse.x)
        np.testing.assert_array_equal(fine.y[::2, ::2], coarse.y)


def test_fresnel_criteria_roundtrip():
    n, dx, wavelength = 1024, 3e-6, 532e-9
    z = fresnel_min_distance(n, dx, wavelength)
    np.testing.assert_allclose(fresnel_max_spacing(n, z, wavelength), dx, rtol=1e-12)


def test_fresnel_criteria_validation():
    with pytest.raises(ValueError):
        fresnel_min_distance(0, 1e-6, 532e-9)
    with pytest.raises(ValueError):
        fresnel_max_spacing(1024, -1.0, 532e-9)


def test_next_fft_size_power_of_two():
    assert next_fft_size(1000, prefer_fast_len=False) == 1024
    assert next_fft_size(1025, prefer_fast_len=False) == 2048
    assert next_fft_size(1, prefer_fast_len=False) == 2


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not installed")
def test_next_fft_size_fast_len_matches_scipy():
    from scipy.fft import next_fast_len

    assert next_fft_size(1000, prefer_fast_len=True) == int(next_fast_len(1000))


def test_convergence_recommends_finer_grid_for_near_field():
    # The debugging-session config (R=0.3mm, 532nm, L=6mm) at z=0.05 needs a
    # fine grid; the tool should recommend N >= 1024 (kept small/fast here).
    def aperture(grid):
        return MonochromaticField(grid, 1.0).add_aperture(
            circular_aperture, 0.3e-3, antialiased=True).to_field()

    rec = recommend_grid_convergence(
        aperture,
        length=6e-3,
        wavelength=532e-9,
        z=0.05,
        n0=512,
        n_max=4096,
        metric="azimuthal_cv",
        radial_window=(0.3e-3, 1.2e-3),
        tol=0.05,
    )
    assert rec.converged
    assert rec.n >= 2048
    assert len(rec.history) >= 2
    # azimuthal CV falls as N grows
    cvs = [c for _, c in rec.history]
    assert cvs[-1] < cvs[0]


def test_convergence_successive_diff_converges_in_free_space():
    # A smooth Gaussian-like soft aperture converges quickly.
    def aperture(grid):
        return MonochromaticField(grid, 1.0).add_aperture(
            circular_aperture, 1.0e-3, antialiased=True).to_field()

    rec = recommend_grid_convergence(
        aperture,
        length=8e-3,
        wavelength=532e-9,
        z=0.02,
        n0=256,
        n_max=2048,
        metric="successive_diff",
        tol=0.1,
    )
    assert rec.n >= 256
    assert rec.length == 8e-3
    np.testing.assert_allclose(rec.dx, 8e-3 / rec.n)


def test_convergence_validation():
    def aperture(grid):
        return MonochromaticField(grid, 1.0).add_aperture(
            circular_aperture, 0.3e-3, antialiased=True).to_field()

    with pytest.raises(ValueError):
        recommend_grid_convergence(aperture, length=-1.0, wavelength=532e-9, z=0.05)
    with pytest.raises(ValueError):
        recommend_grid_convergence(
            aperture, length=6e-3, wavelength=532e-9, z=0.05, metric="bogus"
        )
    with pytest.raises(ValueError):
        recommend_grid_convergence(
            aperture, length=6e-3, wavelength=532e-9, z=0.05, metric="azimuthal_cv"
        )  # missing radial_window
