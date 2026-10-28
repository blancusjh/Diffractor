"""Analytic and energy checks for the real, damped scalar wave equation."""

import numpy as np
import pytest

from diffractor.mathutils.grids import Grid
from diffractor.physics.time_domain import WaveEquation2D


def _grid(nx=49, ny=37, dx=0.13, dy=0.2):
    return Grid(*np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy))


def test_rectangular_grid_mode_matches_discrete_dispersion_and_energy():
    grid = _grid()
    index = 1.5
    solver = WaveEquation2D(grid, refractive_index=index, c0=1)
    nx, ny = grid.shape[1], grid.shape[0]
    mx, my = 2, 3
    mode = np.sin(mx * np.pi * grid.x / grid.x[0, -1]) * np.sin(
        my * np.pi * grid.y / grid.y[-1, 0]
    )
    eigenvalue = 4 * (
        np.sin(mx * np.pi / (2 * (nx - 1))) ** 2 / solver.dx**2
        + np.sin(my * np.pi / (2 * (ny - 1))) ** 2 / solver.dy**2
    )
    # Exact five-point/centered-time dispersion for a Dirichlet sine mode.
    frequency = 2 * np.arcsin(solver.dt * np.sqrt(eigenvalue) / (2 * index)) / solver.dt
    solver.reset(mode)
    initial_energy = solver.energy()
    solver.step(400)
    expected = mode * np.cos(frequency * solver.time)
    np.testing.assert_allclose(solver.u, expected, atol=3e-13, rtol=0)
    assert abs(solver.energy() / initial_energy - 1) < 2e-13


def test_wave_speed_converges_to_continuum_at_second_order():
    errors = []
    for refinement, intervals in enumerate((24, 48, 96)):
        grid = _grid(intervals + 1, intervals + 1, 1 / intervals, 1 / intervals)
        solver = WaveEquation2D(grid, refractive_index=1.4, c0=1)
        mode = np.sin(np.pi * grid.x) * np.sin(2 * np.pi * grid.y)
        solver.reset(mode)
        solver.step(6 * 2**refinement)
        continuum_frequency = np.pi * np.sqrt(5) / 1.4
        continuum = mode * np.cos(continuum_frequency * solver.time)
        errors.append(float(np.max(abs(solver.u - continuum))))
    # Both space and time are refined together at a fixed Courant fraction.
    # The continuum comparison includes propagation rather than recalculating
    # an expected field using the finite-difference update under test.
    assert 3.5 < errors[0] / errors[1] < 4.5
    assert 3.5 < errors[1] / errors[2] < 4.5


def test_zero_state_mean_square_forcing_units_and_reset():
    solver = WaveEquation2D(_grid(), c0=1)
    solver.step(20)
    assert np.count_nonzero(solver.u) == 0
    assert np.count_nonzero(solver.mean_square) == 0
    assert solver.energy() == 0

    solver.reset()
    calls = []

    def source(time):
        calls.append(time)
        return 3.0

    solver.step(forcing=source)
    np.testing.assert_allclose(solver.u[1:-1, 1:-1], 0.5 * 3 * solver.dt**2)
    np.testing.assert_allclose(solver.mean_square, solver.u**2)
    first_square = solver.u**2
    solver.step(forcing=source)
    np.testing.assert_allclose(solver.mean_square, (first_square + solver.u**2) / 2)
    np.testing.assert_allclose(calls, [0, solver.dt])
    assert solver.time == 2 * solver.dt and solver.step_count == 2
    solver.reset()
    assert solver.time == 0 and solver.step_count == 0
    assert (
        not solver.u.any()
        and not solver.u_previous.any()
        and not solver.mean_square.any()
    )


def test_nonzero_initial_velocity_has_second_order_startup():
    solver = WaveEquation2D(_grid(), damping=0.3, c0=1)
    solver.reset(initial_velocity=2.0)
    # Far from the boundary, Laplacian(u0)=0, v0=2 and a0=-2*sigma*v0.
    solver.step()
    expected = 2 * solver.dt - 0.5 * (2 * 0.3 * 2) * solver.dt**2
    # The centered damping update differs from the Taylor polynomial by O(dt³).
    assert abs(solver.u[10, 10] - expected) < 2 * solver.dt**3


def test_heterogeneous_medium_and_damping_preserve_discrete_energy_balance():
    grid = _grid(67, 45, 0.1, 0.15)
    index = np.where(grid.x < 3, 1.0, 1.7)
    initial = np.exp(-((grid.x - 2.8) ** 2 + (grid.y - 3.0) ** 2) / 0.7)
    for damping in (0.0, 0.25):
        solver = WaveEquation2D(grid, index, damping=damping, c0=1)
        solver.reset(initial)
        energies = [solver.energy()]
        for _ in range(250):
            solver.step()
            energies.append(solver.energy())
        assert np.isfinite(solver.u).all()
        assert min(energies) > 0
        if damping == 0:
            assert np.max(abs(np.array(energies) / energies[0] - 1)) < 2e-13
        else:
            assert np.max(np.diff(energies)) < 2e-13 * energies[0]
            assert energies[-1] < 0.1 * energies[0]


def test_solid_mask_is_a_fixed_reflector_not_a_loss_layer():
    grid = _grid(61, 41, 0.1, 0.15)
    solid = np.zeros(grid.shape, dtype=bool)
    solid[:, 30] = True
    solver = WaveEquation2D(grid, solid_mask=solid, c0=1)
    initial = np.exp(-((grid.x - 1.5) ** 2 + (grid.y - 3) ** 2) / 0.2)
    initial[:, 30:] = 0
    solver.reset(initial)
    energy = solver.energy()
    solver.step(220)
    assert not solver.u[:, 30:].any()
    assert not solver.u[[0, -1], :].any() and not solver.u[:, [0, -1]].any()
    assert abs(solver.energy() / energy - 1) < 2e-13


def _right_moving_packet(grid, *, wavelength, width):
    """A nearly normal-incidence Gaussian packet and its initial velocity."""
    coordinate = grid.x - 50
    wavenumber = 2 * np.pi / wavelength
    envelope = np.exp(-0.5 * (coordinate / width) ** 2)
    # A transverse Dirichlet eigenmode avoids edge transients from clipping
    # a uniform plane wave. Its transverse wavenumber is small vs the carrier.
    transverse = np.sin(np.pi * grid.y / 32)
    initial = envelope * np.cos(wavenumber * coordinate) * transverse
    velocity = (
        envelope
        * (
            coordinate / width**2 * np.cos(wavenumber * coordinate)
            + wavenumber * np.sin(wavenumber * coordinate)
        )
        * transverse
    )
    return initial, velocity


def test_index_interface_reflects_and_transmits_scalar_fresnel_amplitudes():
    grid = _grid(961, 33, 0.25, 1.0)
    x = grid.x[0]
    initial, velocity = _right_moving_packet(grid, wavelength=6, width=12)
    index = np.where(grid.x < 120, 1.0, 1.5)
    solver = WaveEquation2D(grid, index, c0=1)
    solver.reset(initial, velocity)
    solver.step(round(140 / solver.dt))
    reflected = solver.u[16, (x > 20) & (x < 90)]
    transmitted = solver.u[16, (x > 140) & (x < 200)]

    # Continuity of scalar Ez and its normal derivative at constant mu gives
    # r=(n1-n2)/(n1+n2)=-0.2, t=2*n1/(n1+n2)=0.8 at normal incidence.
    # The carrier is resolved by 24 samples/vacuum wavelength and 16 samples
    # in glass. A 1.5% absolute-amplitude tolerance covers discrete dispersion,
    # sampling of the packet maximum, and the small transverse wave number
    # ky/k=6/(2*32), which changes the ideal r by about 0.1% in amplitude.
    assert abs(np.max(abs(reflected)) - 0.2) < 0.015
    assert abs(np.max(abs(transmitted)) - 0.8) < 0.015

    # Use an independently propagated incident packet only to locate its
    # mirrored phase: the analytic sign of the reflected coefficient is minus.
    incident = WaveEquation2D(grid, c0=1)
    incident.reset(initial, velocity)
    incident.step(solver.step_count)
    left = np.flatnonzero((x > 20) & (x < 90))
    mirror = incident.u[16, x.size - 1 - left]
    signed_coefficient = float(reflected @ mirror / (mirror @ mirror))
    assert abs(signed_coefficient + 0.2) < 0.015


def test_graded_sponge_reduces_a_measured_boundary_return():
    grid = _grid(961, 33, 0.25, 1.0)
    initial, velocity = _right_moving_packet(grid, wavelength=4, width=10)
    # A 60-m (15 carrier wavelength) cubic layer, with sigma in inverse
    # seconds. This tests one smooth sponge configuration, not a PML claim.
    ramp = np.clip((grid.x - 180) / 60, 0, 1)
    damping = 0.5 * ramp**3
    reflected_peaks = []
    for sigma in (0.0, damping):
        solver = WaveEquation2D(grid, damping=sigma, c0=1)
        solver.reset(initial, velocity)
        solver.step(round(350 / solver.dt))
        observation = (grid.x[0] > 20) & (grid.x[0] < 140)
        reflected_peaks.append(float(np.max(abs(solver.u[16, observation]))))
    assert reflected_peaks[0] > 0.9
    assert reflected_peaks[1] < 0.01 * reflected_peaks[0]


def test_cfl_inputs_regular_lattice_and_static_medium_validation():
    grid = _grid()
    solver = WaveEquation2D(grid, refractive_index=0.8, c0=1)
    expected_limit = 0.8 / np.hypot(1 / solver.dx, 1 / solver.dy)
    np.testing.assert_allclose(solver.cfl_limit, expected_limit, rtol=1e-15)
    with pytest.raises(ValueError, match="CFL"):
        WaveEquation2D(grid, refractive_index=0.8, c0=1, dt=expected_limit * 1.001)
    for value in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            WaveEquation2D(grid, refractive_index=value)
    for value in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            WaveEquation2D(grid, damping=value)
    for value in (0, -1, np.nan, np.inf, 1):
        with pytest.raises(ValueError):
            WaveEquation2D(grid, courant=value)
    distorted = grid.x.copy()
    distorted[:, 5] += 0.01 * solver.dx
    with pytest.raises(ValueError, match="regular Cartesian"):
        WaveEquation2D(Grid(distorted, grid.y))
    with pytest.raises(ValueError, match="shape"):
        solver.step(forcing=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        solver.step(forcing=lambda time: np.nan)
    with pytest.raises(ValueError, match="real"):
        solver.reset(initial_field=1j)
    with pytest.raises(ValueError, match="steps"):
        solver.step(1.5)
    with pytest.raises(ValueError):
        solver.refractive_index[5, 5] = 2
    with pytest.raises(ValueError):
        solver.damping[5, 5] = 1
