"""Source physics, scene geometry and headless functional controls."""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from diffractor.physics.wave_scene import (
    WaveScene,
    WaveSource,
    double_slit_scene,
    finite_line_source,
    lens_scene,
    sponge_damping,
    wave_grid,
)
from diffractor.viz.wave_simulation import WaveController, WaveViewer, export_wave_scene


def small_scene():
    wavelength = 532e-9
    grid = wave_grid(wavelength, width=8, height=6)
    source = finite_line_source(grid, wavelength, x=-2, half_height=1.5)
    return WaveScene(
        name="test",
        description="A small driven wave",
        grid=grid,
        refractive_index=np.ones(grid.shape),
        damping=sponge_damping(grid, wavelength, thickness=1),
        solid_mask=np.zeros(grid.shape, dtype=bool),
        sources=(source,),
        overlays=(),
        wavelength=wavelength,
    )


def test_lens_is_a_volume_index_and_resolves_local_wavelength():
    scene = lens_scene()
    unit = scene.wavelength
    center_row = np.argmin(np.abs(scene.grid.y[:, 0]))
    material = scene.refractive_index[center_row] > 1
    width = np.ptp(scene.grid.x[center_row, material])
    dx, _ = scene.grid.spacing
    assert abs(width / unit - 4) < 2 * dx / unit
    assert np.max(scene.refractive_index) == 1.5
    assert scene.wavelength / (dx * np.max(scene.refractive_index)) >= 12 - 1e-9
    assert not np.any(scene.solid_mask)
    assert scene.maximum_frequency_scale == pytest.approx(1.0)


def test_double_slit_uses_reflecting_cells_with_openings():
    scene = double_slit_scene()
    column = int(np.argmin(np.abs(scene.grid.x[0] / scene.wavelength + 8)))
    y = scene.grid.y[:, 0] / scene.wavelength
    assert scene.solid_mask[np.argmin(np.abs(y)), column]
    assert not scene.solid_mask[np.argmin(np.abs(y - 4)), column]
    assert not scene.solid_mask[np.argmin(np.abs(y + 4)), column]
    assert np.all(scene.forcing(1 / scene.frequency)[scene.solid_mask] == 0)


def test_sources_superpose_with_ramped_physical_acceleration():
    scene = small_scene()
    source = scene.sources[0]
    second = replace(source, amplitude=0.5)
    combined = replace(scene, sources=(source, second))
    time = 4.25 / source.frequency
    expected = source.profile * 0.2 * (2 * np.pi * source.frequency) ** 2
    np.testing.assert_allclose(source.acceleration(time), expected, rtol=1e-12)
    np.testing.assert_allclose(
        combined.forcing(time), 1.5 * source.acceleration(time), rtol=1e-12
    )
    assert np.all(source.acceleration(0) == 0)
    ramped = np.max(np.abs(source.acceleration(0.25 / source.frequency)))
    full = np.max(np.abs(source.acceleration(time)))
    assert 0 < ramped < 0.02 * full


def test_pulse_envelope_decays_after_its_emission_window():
    source = replace(small_scene().sources[0], waveform="pulse")
    near_peak = np.max(np.abs(source.acceleration(8.25 / source.frequency)))
    late = np.max(np.abs(source.acceleration(20.25 / source.frequency)))
    assert near_peak > 0
    assert late < 1e-7 * near_peak


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frequency": 0},
        {"ramp_cycles": 0},
        {"pulse_width_cycles": 0},
        {"frequency": np.nan},
        {"amplitude": -1},
        {"waveform": "invalid"},
    ],
)
def test_invalid_source_parameters_raise_clear_errors(kwargs):
    parameters = {"profile": np.ones((3, 3)), "frequency": 1.0}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        WaveSource(**parameters)


@pytest.mark.parametrize(
    "kwargs", [{"width": 0}, {"height": -1}, {"wavelength": np.nan}]
)
def test_invalid_grid_parameters_are_rejected(kwargs):
    parameters = {"wavelength": 532e-9}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        wave_grid(**parameters)


def test_sponge_is_zero_interior_and_increases_toward_edges():
    scene = small_scene()
    center = tuple(size // 2 for size in scene.grid.shape)
    assert scene.damping[center] == 0
    assert np.min(scene.damping) == 0
    assert np.all(scene.damping >= 0)
    assert np.max(scene.damping[0]) > 0


def test_controller_pause_step_source_settings_and_reset():
    controller = WaveController(small_scene(), steps_per_frame=2)
    controller.toggle_pause()
    controller.tick()
    assert controller.solver.step_count == 0
    controller.step()
    assert controller.solver.step_count == 1
    controller.set_pending_source(frequency_scale=0.75, amplitude_scale=0.5)
    assert controller.frequency_scale == 1
    controller.reset()
    assert controller.frequency_scale == 0.75
    assert controller.amplitude_scale == 0.5
    assert controller.solver.step_count == 0
    assert np.all(controller.solver.u == 0)
    assert np.all(controller.solver.mean_square == 0)
    with pytest.raises(ValueError, match="finer grid"):
        controller.set_pending_source(frequency_scale=2)


def test_headless_viewer_controls_do_not_advance_on_initial_draw():
    viewer = WaveViewer(small_scene())
    viewer.figure.canvas.draw()
    viewer.animation.event_source.stop()
    assert viewer.controller.solver.step_count == 0
    viewer._pause()
    viewer._step()
    assert viewer.controller.solver.step_count == 1
    viewer._mode_changed("mean square")
    assert viewer.mode_buttons.value_selected == "mean square"
    viewer._reset()
    assert viewer.controller.solver.step_count == 0
    assert viewer.controller.mode == "mean square"
    plt.close(viewer.figure)


def test_export_frames_are_actual_solver_evolution_and_data_matches(tmp_path):
    scene = small_scene()
    result = export_wave_scene(
        scene,
        tmp_path / "wave",
        frames=3,
        fps=10,
        warmup_periods=1,
        steps_per_frame=2,
        mp4=False,
    )
    assert result["final_step_count"] == result["warmup_steps"] + 4
    assert result["frames_per_optical_period"] >= 12
    with Image.open(result["outputs"]["gif"]) as image:
        assert image.n_frames == 3
    with np.load(result["outputs"]["data"]) as data:
        assert np.max(np.abs(data["field"])) > 0
        assert np.max(data["mean_square"]) > 0


def test_export_sampling_guard_uses_fastest_custom_source(tmp_path):
    scene = small_scene()
    faster = replace(scene.sources[0], frequency=scene.frequency * 1.4)
    custom = replace(scene, sources=(scene.sources[0], faster))
    with pytest.raises(ValueError, match="12 frames"):
        export_wave_scene(
            custom,
            tmp_path / "aliased",
            frames=2,
            steps_per_frame=3,
            warmup_periods=0,
            mp4=False,
        )
