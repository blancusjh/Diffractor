"""Physical acceptance checks for the three complete optical experiments."""

import importlib.util
from pathlib import Path

import numpy as np

from diffractor import sample_intensity


def _example(name):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


coherent_incoherent = _example("coherent_incoherent")
four_f_imaging = _example("four_f_imaging")
two_source_interference = _example("two_source_interference")


def test_crossing_waves_match_analytic_fringe_and_point_probes_at_two_resolutions():
    for samples in (128, 256):
        _, _, profiles, expected, phases, point, _ = two_source_interference.simulate(samples)
        np.testing.assert_allclose(profiles, expected, atol=1e-10)
        np.testing.assert_allclose(np.abs(point)**2, 2 + 2 * np.cos(phases), atol=1e-10)


def test_four_f_inverts_the_object_and_passive_filters_reduce_power():
    source, _, images, transmission, errors = four_f_imaging.simulate(256)
    # Centered FFT coordinates are half-open: inversion is a flip plus one
    # sample of periodic roll, not an ordinary image flip.
    inverted = np.roll(source.intensity[::-1, ::-1], (1, 1), axis=(0, 1))
    np.testing.assert_allclose(images["Unfiltered"].intensity, inverted, atol=1e-10)
    np.testing.assert_allclose(transmission["Unfiltered"], 1, atol=1e-12)
    assert 0 < transmission["Low-pass"] < 1
    assert 0 < transmission["High-pass"] < 1
    assert max(errors.values()) < 1e-10


def test_four_f_sampling_convergence_at_fixed_physical_coordinates():
    x = np.linspace(-1e-3, 1e-3, 301)
    curves, power = [], []
    for samples in (128, 256, 512):
        _, _, images, transmission, _ = four_f_imaging.simulate(samples)
        curves.append(sample_intensity(images["Low-pass"], x, 0.15e-3))
        power.append(transmission["Low-pass"])
    coarse_error = np.sqrt(np.mean((curves[0] - curves[1])**2))
    fine_error = np.sqrt(np.mean((curves[1] - curves[2])**2))
    assert fine_error < 0.7 * coarse_error
    assert fine_error < 1e-3  # relative to unit incident intensity
    assert abs(power[2] - power[1]) < 1e-6


def test_random_phase_ensemble_converges_to_incoherent_focal_pattern():
    locked, incoherent, averaged, phases, gamma, _, error = coherent_incoherent.simulate(
        samples=128, realizations=1024, seed=7)
    np.testing.assert_allclose(phases, np.random.default_rng(7).uniform(0, 2 * np.pi, 1024))
    assert abs(gamma) < 0.05
    assert np.max(np.abs(averaged - incoherent)) / incoherent.max() < 0.05
    # The center is constructive for locked equal emitters; each emitter
    # contributes half the incoherent peak there.
    center = (64, 64)
    np.testing.assert_allclose(locked.intensity[center] / incoherent[center], 2, atol=1e-12)
    assert error < 1e-12


def test_two_gaussian_emitters_match_the_analytic_focal_envelope():
    exp = coherent_incoherent
    for samples in (128, 256):
        locked, incoherent, *_ = exp.simulate(samples, realizations=4)
        x, y = locked.grid
        amplitude = np.pi * exp.WAIST**2 / (exp.WAVELENGTH * exp.FOCAL_LENGTH)
        single = amplitude**2 * np.exp(
            -2 * (np.pi * exp.WAIST / (exp.WAVELENGTH * exp.FOCAL_LENGTH))**2 * (x**2 + y**2))
        fringes = 4 * single * np.cos(np.pi * exp.SEPARATION * x / (exp.WAVELENGTH * exp.FOCAL_LENGTH))**2
        np.testing.assert_allclose(incoherent, 2 * single, atol=1e-12)
        np.testing.assert_allclose(locked.intensity, fringes, atol=1e-12)
