"""Interactive controls and real solver-stepped exports for wave scenes."""

from __future__ import annotations

import time
from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, RadioButtons, Slider

from ..physics.wave_scene import WaveScene

BACKGROUND = "#07090d"
TEXT = "#d6dbe3"
MODES = ("luminous", "signed", "mean square")


def _integer_at_least(value, minimum, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer of at least {minimum}.")


class WaveController:
    """UI-independent simulation state, controls and source parameters."""

    def __init__(self, scene: WaveScene, *, steps_per_frame=2):
        _integer_at_least(steps_per_frame, 1, "steps_per_frame")
        self.scene = scene
        self.solver = scene.make_solver()
        self.steps_per_frame = steps_per_frame
        self.paused = False
        self.mode = "luminous"
        self.gain = 1.0
        self.frequency_scale = 1.0
        self.amplitude_scale = 1.0
        self.pending_frequency_scale = 1.0
        self.pending_amplitude_scale = 1.0

    def _forcing(self, time):
        return self.scene.forcing(
            time,
            frequency_scale=self.frequency_scale,
            amplitude_scale=self.amplitude_scale,
        )

    def step(self, count=1):
        """Advance numerical time even while playback is paused."""
        return self.solver.step(count, forcing=self._forcing)

    def tick(self):
        if not self.paused:
            self.step(self.steps_per_frame)

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused

    def set_pending_source(self, *, frequency_scale=None, amplitude_scale=None):
        frequency = (
            self.pending_frequency_scale if frequency_scale is None else frequency_scale
        )
        amplitude = (
            self.pending_amplitude_scale if amplitude_scale is None else amplitude_scale
        )
        # Validate the parameters before storing them; no PDE step is taken.
        self.scene.forcing(0, frequency_scale=frequency, amplitude_scale=amplitude)
        self.pending_frequency_scale = float(frequency)
        self.pending_amplitude_scale = float(amplitude)

    def reset(self):
        """Apply pending source settings and clear all accumulated state."""
        self.frequency_scale = self.pending_frequency_scale
        self.amplitude_scale = self.pending_amplitude_scale
        self.solver.reset()

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError("Unknown display mode.")
        self.mode = mode

    def set_gain(self, gain):
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("gain must be finite and positive.")
        self.gain = float(gain)

    def display_data(self):
        """Map fields using fixed references, never frame normalization."""
        if self.mode == "mean square":
            scaled = self.gain * self.solver.mean_square / 0.25
            return np.clip(scaled, 0, 1) ** 0.45
        scaled = np.clip(self.gain * self.solver.u, -1, 1)
        magnitude = np.abs(scaled) ** 0.65
        return np.sign(scaled) * magnitude if self.mode == "signed" else magnitude


def _colormap(mode):
    if mode == "signed":
        colors = [
            "#d8f5ff",
            "#3e8bc0",
            "#112238",
            BACKGROUND,
            "#34100c",
            "#e76520",
            "#fff1cd",
        ]
    else:
        colors = [
            BACKGROUND,
            "#230c13",
            "#6b1824",
            "#bc3a1f",
            "#f68c34",
            "#ffdc9a",
            "#fff9e7",
        ]
    return LinearSegmentedColormap.from_list("scalar_wave_" + mode, colors)


class WaveDisplay:
    """A reusable spatial view; controls and file writers use the same artist."""

    def __init__(self, controller, *, controls=False):
        self.controller = controller
        scene = controller.scene
        unit = scene.wavelength
        dx, dy = scene.grid.spacing
        x, y = scene.grid
        with plt.rc_context({"font.family": "serif"}):
            self.figure = plt.figure(
                figsize=(12.8, 8 if controls else 7.2), facecolor=BACKGROUND
            )
            bounds = (
                [0.065, 0.31, 0.91, 0.58] if controls else [0.065, 0.15, 0.91, 0.72]
            )
            self.ax = self.figure.add_axes(bounds, facecolor=BACKGROUND)
            self.image = self.ax.imshow(
                controller.display_data(),
                origin="lower",
                aspect="equal",
                extent=[
                    (x.min() - dx / 2) / unit,
                    (x.max() + dx / 2) / unit,
                    (y.min() - dy / 2) / unit,
                    (y.max() + dy / 2) / unit,
                ],
                cmap=_colormap(controller.mode),
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            for overlay in scene.overlays:
                self.ax.plot(
                    overlay.x / unit,
                    overlay.y / unit,
                    color=overlay.color,
                    lw=1.2,
                    alpha=0.9,
                )
            self.ax.set_xlabel("x / λ₀", color=TEXT)
            self.ax.set_ylabel("y / λ₀", color=TEXT)
            self.ax.tick_params(colors="#929dab", labelsize=9)
            for spine in self.ax.spines.values():
                spine.set_color("#343b45")
            self.figure.text(0.065, 0.94, scene.description, color=TEXT, fontsize=17)
            self.mode_label = self.figure.text(
                0.065, 0.90, "", color="#aab3c0", fontsize=10
            )
            self.time_label = self.figure.text(
                0.975, 0.90, "", color=TEXT, fontsize=10, ha="right"
            )
            self.figure.text(
                0.065,
                0.045,
                "Scalar time-domain PDE · spatial index and reflections · "
                "graded absorbing boundary",
                color="#8f99a8",
                fontsize=9,
            )
            self.figure.text(
                0.065,
                0.022,
                "Mean square is ⟨u²⟩ over elapsed numerical steps; "
                "it is not electromagnetic intensity.",
                color="#707c8c",
                fontsize=9,
            )
        self.update()

    def update(self):
        controller = self.controller
        mode = controller.mode
        self.image.set_data(controller.display_data())
        self.image.set_cmap(_colormap(mode))
        self.image.set_clim(-1 if mode == "signed" else 0, 1)
        labels = {
            "luminous": "Luminous instantaneous field |u| · display γ = 0.65",
            "signed": (
                "Signed instantaneous field u · blue − / amber + · display γ = 0.65"
            ),
            "mean square": "Running mean square ⟨u²⟩ · display γ = 0.45",
        }
        self.mode_label.set_text(labels[mode] + f" · gain {controller.gain:g}")
        periods = controller.solver.time * controller.scene.frequency
        self.time_label.set_text(
            f"t / T₀ = {periods:.2f} · step {controller.solver.step_count}"
        )
        return self.image, self.mode_label, self.time_label


class WaveViewer:
    """Matplotlib play/pause, one-step, reset, mode and source controls."""

    def __init__(self, scene, *, steps_per_frame=2, mode="luminous"):
        self.controller = WaveController(scene, steps_per_frame=steps_per_frame)
        self.controller.set_mode(mode)
        self.display = WaveDisplay(self.controller, controls=True)
        self.figure = self.display.figure
        self._widgets = []
        self._build_controls()
        self.figure.canvas.mpl_connect("key_press_event", self._key_pressed)
        self.animation = FuncAnimation(
            self.figure,
            self._animate,
            interval=40,
            init_func=self.display.update,
            blit=False,
            cache_frame_data=False,
        )

    def _button(self, bounds, label, callback):
        button = Button(
            self.figure.add_axes(bounds), label, color="#cbd1d8", hovercolor="#f2d29a"
        )
        button.on_clicked(callback)
        self._widgets.append(button)
        return button

    def _build_controls(self):
        self.pause_button = self._button(
            [0.065, 0.20, 0.10, 0.045], "Pause", self._pause
        )
        self._button([0.175, 0.20, 0.10, 0.045], "One step", self._step)
        self._button([0.285, 0.20, 0.10, 0.045], "Reset", self._reset)
        self.mode_buttons = RadioButtons(
            self.figure.add_axes([0.80, 0.11, 0.17, 0.14]),
            MODES,
            active=MODES.index(self.controller.mode),
        )
        self.mode_buttons.on_clicked(self._mode_changed)
        self._widgets.append(self.mode_buttons)

        self.gain_slider = Slider(
            self.figure.add_axes([0.48, 0.21, 0.23, 0.025]),
            "Gain",
            0.1,
            4.0,
            valinit=1.0,
        )
        self.gain_slider.on_changed(self._gain_changed)
        maximum = min(1.5, self.controller.scene.maximum_frequency_scale)
        self.frequency_slider = Slider(
            self.figure.add_axes([0.14, 0.12, 0.22, 0.025]),
            "Frequency ×",
            0.5,
            maximum,
            valinit=1.0,
        )
        self.amplitude_slider = Slider(
            self.figure.add_axes([0.51, 0.12, 0.22, 0.025]),
            "Drive ×",
            0.0,
            2.0,
            valinit=1.0,
        )
        self.frequency_slider.on_changed(self._source_changed)
        self.amplitude_slider.on_changed(self._source_changed)
        self._widgets.extend(
            (self.gain_slider, self.frequency_slider, self.amplitude_slider)
        )
        for slider in (self.gain_slider, self.frequency_slider, self.amplitude_slider):
            slider.label.set_color(TEXT)
            slider.valtext.set_color(TEXT)
        self.figure.text(
            0.14,
            0.082,
            "Frequency and drive changes apply on Reset. "
            "Space: pause · N: step · R: reset",
            color="#a4adbb",
            fontsize=9,
        )

    def _redraw(self):
        self.display.update()
        self.figure.canvas.draw_idle()

    def _pause(self, _event=None):
        paused = self.controller.toggle_pause()
        self.pause_button.label.set_text("Play" if paused else "Pause")
        self._redraw()

    def _step(self, _event=None):
        self.controller.step(1)
        self._redraw()

    def _reset(self, _event=None):
        self.controller.reset()
        self._redraw()

    def _mode_changed(self, mode):
        self.controller.set_mode(mode)
        if self.mode_buttons.value_selected != mode:
            self.mode_buttons.set_active(MODES.index(mode))
        self._redraw()

    def _gain_changed(self, gain):
        self.controller.set_gain(gain)
        self._redraw()

    def _source_changed(self, _value):
        self.controller.set_pending_source(
            frequency_scale=self.frequency_slider.val,
            amplitude_scale=self.amplitude_slider.val,
        )

    def _key_pressed(self, event):
        callbacks = {" ": self._pause, "n": self._step, "r": self._reset}
        if event.key in callbacks:
            callbacks[event.key]()

    def _animate(self, _frame):
        self.controller.tick()
        return self.display.update()

    def show(self):
        plt.show()


def export_wave_scene(
    scene,
    output_prefix,
    *,
    frames=64,
    fps=20,
    steps_per_frame=2,
    warmup_periods=50.0,
    mode="luminous",
    mp4=True,
):
    """Export actual numerical evolution; the solver advances between frames."""
    _integer_at_least(frames, 2, "frames")
    _integer_at_least(fps, 1, "fps")
    if (
        isinstance(warmup_periods, (bool, np.bool_))
        or not np.isfinite(warmup_periods)
        or warmup_periods < 0
    ):
        raise ValueError("warmup_periods must be finite and nonnegative.")
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    controller = WaveController(scene, steps_per_frame=steps_per_frame)
    controller.set_mode(mode)
    fastest_frequency = max(source.frequency for source in scene.sources)
    numerical_phase_step = steps_per_frame * controller.solver.dt * fastest_frequency
    if numerical_phase_step > 1 / 12 * (1 + 1e-9):
        raise ValueError(
            "Export needs at least 12 frames per fastest driven optical period."
        )

    started = time.perf_counter()
    warmup_steps = round(warmup_periods / (controller.solver.dt * scene.frequency))
    controller.step(warmup_steps)
    warmup_seconds = time.perf_counter() - started
    display = WaveDisplay(controller)
    output = {"gif": str(output_prefix.with_suffix(".gif"))}
    gif_writer = PillowWriter(fps=fps)
    mp4_available = mp4 and writers.is_available("ffmpeg")
    mp4_writer = (
        FFMpegWriter(fps=fps, codec="libx264", bitrate=3000) if mp4_available else None
    )
    if mp4_available:
        output["mp4"] = str(output_prefix.with_suffix(".mp4"))

    started = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(gif_writer.saving(display.figure, output["gif"], dpi=80))
        if mp4_writer is not None:
            stack.enter_context(
                mp4_writer.saving(display.figure, output["mp4"], dpi=100)
            )
        for frame in range(frames):
            if frame > 0:
                controller.step(steps_per_frame)
            display.update()
            gif_writer.grab_frame(facecolor=BACKGROUND)
            if mp4_writer is not None:
                mp4_writer.grab_frame(facecolor=BACKGROUND)
    rendering_seconds = time.perf_counter() - started

    poster_path = output_prefix.with_suffix(".png")
    display.figure.savefig(poster_path, dpi=100, facecolor=BACKGROUND)
    output["poster"] = str(poster_path)
    controller.set_mode("mean square")
    display.update()
    mean_path = output_prefix.with_name(
        output_prefix.name + "_mean_square"
    ).with_suffix(".png")
    display.figure.savefig(mean_path, dpi=100, facecolor=BACKGROUND)
    output["mean_square"] = str(mean_path)
    plt.close(display.figure)

    data_path = output_prefix.with_suffix(".npz")
    np.savez_compressed(
        data_path,
        field=controller.solver.u,
        mean_square=controller.solver.mean_square,
        refractive_index=scene.refractive_index,
        damping=scene.damping,
        solid_mask=scene.solid_mask,
        x=scene.grid.x[0],
        y=scene.grid.y[:, 0],
    )
    output["data"] = str(data_path)
    return {
        "outputs": output,
        "frames": frames,
        "fps": fps,
        "warmup_steps": warmup_steps,
        "steps_between_frames": steps_per_frame,
        "final_step_count": controller.solver.step_count,
        "final_time_seconds": controller.solver.time,
        "dt_seconds": controller.solver.dt,
        "cfl_limit_seconds": controller.solver.cfl_limit,
        "optical_period_seconds": 1 / scene.frequency,
        "fastest_source_period_seconds": 1 / fastest_frequency,
        "frames_per_optical_period": 1 / numerical_phase_step,
        "warmup_seconds_wall_clock": warmup_seconds,
        "export_seconds_wall_clock": rendering_seconds,
        "mp4_status": "saved" if mp4_available else "disabled or ffmpeg unavailable",
        "final_peak_field": float(np.max(np.abs(controller.solver.u))),
        "final_peak_mean_square": float(np.max(controller.solver.mean_square)),
    }
