import numpy as np
import pytest

from diffraction import (
    Field,
    fit_zernikes,
    make_grid,
    marechal_strehl,
    noll_to_nm,
    pv,
    rms,
    synthesize_zernikes,
    zernike,
    zernike_name,
)


def test_noll_index_map():
    # Canonical Noll assignments.
    assert noll_to_nm(1) == (0, 0)  # piston
    assert noll_to_nm(2) == (1, 1)  # tip
    assert noll_to_nm(3) == (1, -1)  # tilt
    assert noll_to_nm(4) == (2, 0)  # defocus
    assert noll_to_nm(11) == (4, 0)  # primary spherical
    assert noll_to_nm(22) == (6, 0)  # secondary spherical
    with pytest.raises(ValueError):
        noll_to_nm(0)


def test_known_polynomials():
    rho = np.array([0.0, 0.5, 1.0])
    theta = np.zeros_like(rho)
    # Defocus: sqrt(3) (2 rho^2 - 1)
    np.testing.assert_allclose(zernike(4, rho, theta), np.sqrt(3) * (2 * rho**2 - 1))
    # Primary spherical: sqrt(5) (6 rho^4 - 6 rho^2 + 1)
    np.testing.assert_allclose(
        zernike(11, rho, theta), np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1)
    )


def test_orthonormality_over_disk():
    # RMS-normalized modes: <Zi Zj> over the disk = delta_ij.
    grid = make_grid(512, 2.0)
    x, y = grid
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    inside = rho <= 1.0

    js = [1, 2, 4, 5, 7, 11]
    Z = [zernike(j, rho, theta)[inside] for j in js]
    for a in range(len(js)):
        for b in range(len(js)):
            inner = np.mean(Z[a] * Z[b])
            np.testing.assert_allclose(inner, 1.0 if a == b else 0.0, atol=5e-3)


def test_fit_recovers_synthesized_coefficients():
    grid = make_grid(256, 2.4)
    radius = 1.0
    true = np.zeros(15)
    true[3] = 0.7  # defocus
    true[7] = -0.2  # coma
    true[10] = 0.35  # spherical

    W = synthesize_zernikes(true, grid, radius)
    assert isinstance(W, Field)
    fitted = fit_zernikes(W, radius, jmax=15)
    np.testing.assert_allclose(fitted, true, atol=1e-8)


def test_quartic_decomposition():
    # rho^4 = (1/3) Z1 + 1/(2 sqrt(3)) Z4 + 1/(6 sqrt(5)) Z11 exactly.
    grid = make_grid(512, 2.2)
    x, y = grid
    W = (x**2 + y**2) ** 2
    W[np.sqrt(x**2 + y**2) > 1.0] = np.nan

    c = fit_zernikes(Field(grid, W), 1.0, jmax=15)
    np.testing.assert_allclose(c[0], 1.0 / 3.0, atol=1e-4)
    np.testing.assert_allclose(c[3], 1.0 / (2.0 * np.sqrt(3)), atol=1e-4)
    np.testing.assert_allclose(c[10], 1.0 / (6.0 * np.sqrt(5)), atol=1e-4)
    others = np.delete(c, [0, 3, 10])
    np.testing.assert_allclose(others, 0.0, atol=1e-4)


def test_metrics():
    W = np.array([[0.0, 1.0], [2.0, np.nan]])
    assert pv(W) == 2.0
    np.testing.assert_allclose(rms(W), np.std([0.0, 1.0, 2.0]))
    assert marechal_strehl(0.0) == 1.0
    # lambda/14 RMS -> Strehl ~ 0.8 (the classical diffraction-limited criterion)
    np.testing.assert_allclose(marechal_strehl(1 / 14), 0.817, atol=5e-3)


def test_fit_validation():
    grid = make_grid(64, 2.0)
    W = Field(grid, np.ones((64, 64)))
    with pytest.raises(ValueError):
        fit_zernikes(W, radius=-1.0)
    with pytest.raises(ValueError):
        fit_zernikes(W, radius=1.0, jmax=0)


def test_zernike_name():
    assert zernike_name(4) == "defocus"
    assert zernike_name(11) == "primary spherical"
    assert zernike_name(33) == "Z33"
