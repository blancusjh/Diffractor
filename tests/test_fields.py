import numpy as np
import pytest

from diffraction import gaussian_beam, make_grid, plane_wave


def test_gaussian_beam_profile():
    grid = make_grid(128, 4e-3)
    w0 = 0.5e-3
    field = gaussian_beam(grid, waist=w0)
    U = field.values

    assert field.grid is grid
    assert U.dtype == np.complex128
    np.testing.assert_allclose(U[64, 64], 1.0)  # peak at center
    # 1/e field amplitude at r = w0.
    col = 64 + int(round(w0 / (4e-3 / 128)))
    np.testing.assert_allclose(np.abs(U[64, col]), np.exp(-1.0), rtol=1e-2)

    with pytest.raises(ValueError):
        gaussian_beam(grid, waist=0.0)


def test_plane_wave():
    grid = make_grid(64, 1e-3)
    wavelength = 500e-9

    U = plane_wave(grid, wavelength).values
    np.testing.assert_allclose(U, 1.0)

    # A tilt about y produces a linear phase along x with kx = k sin(θ).
    theta = 1e-3
    U_tilt = plane_wave(grid, wavelength, theta_x=theta).values
    np.testing.assert_allclose(np.abs(U_tilt), 1.0)

    x, _ = grid
    dx = x[0, 1] - x[0, 0]
    phase_step = np.angle(U_tilt[0, 1] / U_tilt[0, 0])
    expected = 2 * np.pi / wavelength * np.sin(theta) * dx
    np.testing.assert_allclose(phase_step, expected, rtol=1e-10)

    with pytest.raises(ValueError):
        plane_wave(grid, wavelength=0.0)
