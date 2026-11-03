"""Reproducible numerical gallery; run with the dev/examples extras installed.

    python examples/validation_gallery.py --output validation-results

All plotted fields are computed by Diffractor. SciPy supplies only the
independent Bessel-function Airy reference. Intensities are in relative units
for a unit-amplitude incident plane wave; no absolute optical power is assumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import scipy
from scipy.special import j1

import diffractor
from diffractor import (
    Grid, circular_aperture, fresnel_zoom_propagator,
    longitudinal_field, make_grid, nslit_aperture, rectangular_aperture, thin_lens,
)

from diffractor.physics.apertures import antialiased

WAVELENGTH = 532e-9
FOCAL_LENGTH = 0.1
DIAMETER = 2e-3
WINDOW = 3.2e-3
SAMPLES = (512, 768)


def source_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def circle_lens(samples):
    grid = make_grid(samples, WINDOW)
    pupil = antialiased(circular_aperture, grid, DIAMETER / 2, factor=8)
    return pupil * thin_lens(grid, FOCAL_LENGTH, WAVELENGTH), pupil


def observation_line(coordinates):
    return Grid(coordinates[None, :], np.zeros((1, coordinates.size)))


def propagate_to_focus(field, grid):
    return fresnel_zoom_propagator(
        field, FOCAL_LENGTH, WAVELENGTH, output_grid=grid,
    )


def airy_profile(x):
    v = np.pi * DIAMETER * np.abs(x) / (WAVELENGTH * FOCAL_LENGTH)
    amplitude = np.ones_like(v)
    np.divide(2 * j1(v), v, out=amplitude, where=v != 0)
    return amplitude ** 2


def first_minimum(x, intensity, radius):
    candidates = np.flatnonzero((x > 0.75 * radius) & (x < 1.25 * radius))
    k = int(candidates[np.argmin(intensity[candidates])])
    # Subsample the local minimum with a three-point parabola.
    left, center, right = intensity[k - 1:k + 2]
    denominator = left - 2 * center + right
    offset = 0.5 * (left - right) / denominator if denominator > 0 else 0.0
    return float(x[k] + np.clip(offset, -1, 1) * (x[1] - x[0]))


def double_slit_lens(samples):
    grid = make_grid(samples, WINDOW)
    slit_width, separation, height = 125e-6, 375e-6, 1e-3
    slits = antialiased(
        nslit_aperture, grid, 2, slit_width, separation, factor=8,
    )
    envelope = antialiased(
        rectangular_aperture, grid, WINDOW, height, factor=8,
    )
    pupil = slits * envelope
    return pupil * thin_lens(grid, FOCAL_LENGTH, WAVELENGTH), {
        "slit_width_m": slit_width, "separation_m": separation, "height_m": height,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation-results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    low, low_pupil = circle_lens(SAMPLES[0])
    high, high_pupil = circle_lens(SAMPLES[1])
    focal_x = np.linspace(-130e-6, 130e-6, 321)
    focal_grid = Grid(*np.meshgrid(focal_x, focal_x))
    focus = propagate_to_focus(high, focal_grid)
    focal_intensity = np.abs(focus.values) ** 2
    mid = focal_x.size // 2
    theoretical_peak = (np.pi * (DIAMETER / 2) ** 2
                        / (WAVELENGTH * FOCAL_LENGTH)) ** 2
    computed = focal_intensity[mid] / theoretical_peak
    low_profile = np.abs(propagate_to_focus(
        low, observation_line(focal_x),
    ).values[0]) ** 2 / theoretical_peak
    analytic = airy_profile(focal_x)
    radius = 3.8317059702075125 * WAVELENGTH * FOCAL_LENGTH / (np.pi * DIAMETER)
    measured_radius = first_minimum(focal_x, computed, radius)

    slit_x = np.linspace(-0.9e-3, 0.9e-3, 1201)
    slit_profiles = []
    for samples in SAMPLES:
        slit_field, slit_geometry = double_slit_lens(samples)
        slit_focus = propagate_to_focus(slit_field, observation_line(slit_x))
        area = 2 * slit_geometry["slit_width_m"] * slit_geometry["height_m"]
        slit_peak = (area / (WAVELENGTH * FOCAL_LENGTH)) ** 2
        slit_profiles.append(np.abs(slit_focus.values[0]) ** 2 / slit_peak)
    frequency = slit_x / (WAVELENGTH * FOCAL_LENGTH)
    slit_theory = (np.sinc(slit_geometry["slit_width_m"] * frequency) ** 2
                   * np.cos(np.pi * slit_geometry["separation_m"] * frequency) ** 2)

    # No per-plane rescaling: preserve the physical growth of intensity at focus.
    zs = np.linspace(0, 0.18, 181)
    section = longitudinal_field(
        low, WAVELENGTH, zs, output_half_width=1.1e-3, output_samples=641,
        pad_factor=2, bandlimit=True, normalize=False,
    )
    global_peak = float(section.intensity.max())
    focus_index = int(np.argmin(np.abs(zs - FOCAL_LENGTH)))
    center_index = int(np.argmin(np.abs(section.t)))
    more_padding = longitudinal_field(
        low, WAVELENGTH, [FOCAL_LENGTH], output_half_width=1.1e-3,
        output_samples=641, pad_factor=3, bandlimit=True, normalize=False,
    )
    padding_change = float(np.max(np.abs(
        section.intensity[focus_index] - more_padding.intensity[0]
    )) / theoretical_peak)
    asm_peak_error = float(abs(
        section.intensity[focus_index, center_index] / theoretical_peak - 1
    ))
    axial_peak_index = np.unravel_index(np.argmax(section.intensity), section.intensity.shape)
    profile_rms = float(np.sqrt(np.mean((computed - analytic) ** 2)))
    slit_rms = float(np.sqrt(np.mean((slit_profiles[1] - slit_theory) ** 2)))
    metrics = {
        "repository_sha": source_revision(),
        "diffractor_version": getattr(diffractor, "__version__", "unknown"),
        "dependencies": {"numpy": np.__version__, "scipy": scipy.__version__,
                         "matplotlib": matplotlib.__version__},
        "wavelength_m": WAVELENGTH,
        "medium_index": 1.0,
        "lens": {
            "focal_length_m": FOCAL_LENGTH, "diameter_m": DIAMETER,
            "input_window_m": WINDOW, "input_samples": list(SAMPLES),
            "pupil_subsamples_per_axis": 8,
            "input_dx_m": [WINDOW / s for s in SAMPLES],
            "fresnel_number_radius_squared_over_lambda_f":
                (DIAMETER / 2) ** 2 / (WAVELENGTH * FOCAL_LENGTH),
            "theoretical_peak_relative_intensity": theoretical_peak,
            "computed_peak_relative_intensity": float(focal_intensity[mid, mid]),
            "peak_relative_error": float(abs(computed[mid] - 1)),
            "theoretical_first_zero_radius_m": radius,
            "computed_first_zero_radius_m": measured_radius,
            "first_zero_relative_error": abs(measured_radius / radius - 1),
            "profile_rms_error_relative_to_theoretical_peak": profile_rms,
            "profile_max_error_relative_to_theoretical_peak":
                float(np.max(np.abs(computed - analytic))),
            "sampling_512_to_768_max_profile_change":
                float(np.max(np.abs(low_profile - computed))),
            "sampled_pupil_area_relative_error_768":
                float(abs(np.sum(high_pupil.values.real) * (WINDOW / SAMPLES[1]) ** 2
                          / (np.pi * (DIAMETER / 2) ** 2) - 1)),
        },
        "double_slit": {
            **slit_geometry, "focal_length_m": FOCAL_LENGTH,
            "theoretical_fringe_period_m":
                WAVELENGTH * FOCAL_LENGTH / slit_geometry["separation_m"],
            "profile_rms_error_relative_to_theoretical_peak": slit_rms,
            "profile_max_error_relative_to_theoretical_peak":
                float(np.max(np.abs(slit_profiles[1] - slit_theory))),
            "sampling_512_to_768_max_profile_change":
                float(np.max(np.abs(slit_profiles[0] - slit_profiles[1]))),
        },
        "longitudinal": {
            "method": "angular spectrum, bandlimit=True, pad_factor=2",
            "input_samples": SAMPLES[0],
            "focus_pad2_to_pad3_max_change_over_theoretical_peak": padding_change,
            "focus_peak_relative_error_vs_paraxial_theory": asm_peak_error,
            "axial_sampling_m": float(zs[1] - zs[0]),
            "focus_note": "The 1 mm axial sampling does not resolve submillimeter diffraction focal shift.",
            "z_range_m": [float(zs[0]), float(zs[-1])], "z_samples": len(zs),
            "transverse_range_m": [-1.1e-3, 1.1e-3], "transverse_samples": 641,
            "global_peak_relative_intensity": global_peak,
            "z_at_sampled_global_peak_m": float(zs[axial_peak_index[0]]),
            "x_at_sampled_global_peak_m": float(section.t[axial_peak_index[1]]),
            "normalization": "one global maximum across all z; never per plane",
        },
        "model": (
            "Scalar monochromatic thin-lens optics. Focal panels use the paraxial "
            "Fresnel integral; the longitudinal panel uses angular spectrum. "
            "Finite sampling and aperture pixel integration limit accuracy. "
            "Relative intensity assumes unit incident amplitude, not calibrated watts."
        ),
    }
    checks = {
        "airy_peak_error_below_1_percent": metrics["lens"]["peak_relative_error"] < 0.01,
        "airy_radius_error_below_1_percent": metrics["lens"]["first_zero_relative_error"] < 0.01,
        "airy_profile_rms_below_0_5_percent": profile_rms < 0.005,
        "airy_sampling_change_below_1_percent":
            metrics["lens"]["sampling_512_to_768_max_profile_change"] < 0.01,
        "slit_profile_rms_below_0_5_percent": slit_rms < 0.005,
        "slit_sampling_change_below_1_percent":
            metrics["double_slit"]["sampling_512_to_768_max_profile_change"] < 0.01,
    }
    checks["asm_focus_padding_change_below_1_percent"] = padding_change < 0.01
    checks["asm_focus_peak_error_below_1_percent"] = asm_peak_error < 0.01
    checks["airy_profile_max_error_below_1_percent"] = (
        metrics["lens"]["profile_max_error_relative_to_theoretical_peak"] < 0.01
    )
    checks["slit_profile_max_error_below_1_percent"] = (
        metrics["double_slit"]["profile_max_error_relative_to_theoretical_peak"] < 0.01
    )
    metrics["checks"] = checks

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "#fbfbf8", "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#5c6570", "text.color": "#182532", "axes.labelcolor": "#182532",
        "xtick.color": "#384958", "ytick.color": "#384958", "savefig.facecolor": "#fbfbf8",
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
    fig.suptitle("Diffractor | computed diffraction and analytic checks",
                 fontsize=17, fontweight="bold")
    a = axes[0, 0]
    im = a.imshow(
        focal_intensity / theoretical_peak, origin="lower",
        extent=[focal_x[0] * 1e6, focal_x[-1] * 1e6,
                focal_x[0] * 1e6, focal_x[-1] * 1e6],
        cmap="magma", norm=LogNorm(vmin=1e-5, vmax=1), interpolation="nearest",
    )
    a.set(title="Circular pupil at the back focal plane", xlabel="x (µm)", ylabel="y (µm)")
    fig.colorbar(im, ax=a, label="I / theoretical on-axis I (log scale)", shrink=0.85)

    a = axes[0, 1]
    im = a.imshow(
        (section.intensity / global_peak).T, origin="lower", aspect="auto",
        extent=[zs[0] * 1e3, zs[-1] * 1e3, section.t[0] * 1e3, section.t[-1] * 1e3],
        cmap="magma", norm=LogNorm(vmin=1e-4, vmax=1), interpolation="nearest",
    )
    a.axvline(FOCAL_LENGTH * 1e3, color="#8ee4ff", ls="--", lw=1, label="geometric focus")
    a.set(title="Lens focusing cone · one global intensity scale",
          xlabel="Distance z (mm)", ylabel="x (mm)")
    a.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.colorbar(im, ax=a, label="I / global maximum (log scale)", shrink=0.85)

    a = axes[1, 0]
    a.semilogy(focal_x * 1e6, np.maximum(computed, 1e-8),
               color="#166a8f", lw=2, label="Diffractor · 768² input")
    a.semilogy(focal_x * 1e6, np.maximum(analytic, 1e-8),
               color="#c46338", lw=1.4, ls="--", label="Airy · [2 J₁(v) / v]²")
    a.axvline(radius * 1e6, color="#73818b", lw=0.9, ls=":")
    a.axvline(-radius * 1e6, color="#73818b", lw=0.9, ls=":")
    a.set(title=f"Airy check · first zero {measured_radius * 1e6:.2f} µm "
                f"(theory {radius * 1e6:.2f})",
          xlabel="x at y = 0 (µm)", ylabel="I / theoretical on-axis I",
          ylim=(1e-5, 1.4), xlim=(focal_x[0] * 1e6, focal_x[-1] * 1e6))
    a.grid(alpha=0.17)
    a.legend(fontsize=8, loc="upper right")

    a = axes[1, 1]
    a.plot(slit_x * 1e3, slit_profiles[1], color="#166a8f", lw=1.8, label="Diffractor")
    a.plot(slit_x * 1e3, slit_theory, color="#c46338", lw=1.2, ls="--",
           label="sinc² envelope × cos² fringes")
    a.set(title="Double slit at the lens focus", xlabel="x at y = 0 (mm)",
          ylabel="I / theoretical on-axis I", ylim=(-0.02, 1.08),
          xlim=(slit_x[0] * 1e3, slit_x[-1] * 1e3))
    a.grid(alpha=0.17)
    a.legend(fontsize=8, loc="upper right")
    fig.supxlabel(
        f"λ = 532 nm · f = 100 mm · circular diameter = 2 mm · "
        f"input convergence 512² → 768² · source {metrics['repository_sha'][:12]}",
        fontsize=9,
    )
    fig.savefig(args.output / "validation_gallery.png", dpi=125)
    plt.close(fig)

    np.savez_compressed(
        args.output / "simulation_data.npz",
        focal_x_m=focal_x, focal_intensity=focal_intensity,
        airy_computed=computed, airy_theory=analytic, airy_low_resolution=low_profile,
        slit_x_m=slit_x, slit_computed=slit_profiles[1], slit_theory=slit_theory,
        slit_low_resolution=slit_profiles[0], longitudinal_z_m=zs,
        longitudinal_x_m=section.t, longitudinal_intensity=section.intensity,
    )
    np.savetxt(
        args.output / "airy_profile.csv",
        np.column_stack([focal_x, computed, analytic, low_profile]),
        delimiter=",", header="x_m,computed_768_over_theoretical_peak,airy_theory,computed_512_over_theoretical_peak",
        comments="",
    )
    np.savetxt(
        args.output / "double_slit_profile.csv",
        np.column_stack([slit_x, slit_profiles[1], slit_theory, slit_profiles[0]]),
        delimiter=",", header="x_m,computed_768_over_theoretical_peak,analytic_theory,computed_512_over_theoretical_peak",
        comments="",
    )
    (args.output / "results.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("Numerical gallery checks failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
