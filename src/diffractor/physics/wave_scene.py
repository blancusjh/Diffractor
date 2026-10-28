"""Composable materials, walls, sources and absorbers for scalar wave scenes.

Lengths and times are SI. Sources prescribe an acceleration drive in field
units per second squared. A graded damping sponge is an approximate absorbing
boundary, not a perfectly matched layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ..mathutils.grids import Grid

LIGHT_SPEED = 299_792_458.0


@dataclass(frozen=True)
class WaveSource:
    """A smooth spatial drive with a ramped sinusoid or Gaussian pulse."""

    profile: np.ndarray
    frequency: float
    amplitude: float = 1.0
    waveform: str = "continuous"
    ramp_cycles: float = 4.0
    pulse_center_cycles: float = 8.0
    pulse_width_cycles: float = 2.0

    def __post_init__(self):
        profile = np.asarray(self.profile)
        if (
            profile.ndim != 2
            or np.iscomplexobj(profile)
            or not np.all(np.isfinite(profile))
        ):
            raise ValueError("profile must be a finite real 2-D array.")
        for name in ("frequency", "ramp_cycles", "pulse_width_cycles"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError("amplitude must be finite and nonnegative.")
        if not np.isfinite(self.pulse_center_cycles) or self.pulse_center_cycles < 0:
            raise ValueError("pulse_center_cycles must be finite and nonnegative.")
        if self.waveform not in ("continuous", "pulse"):
            raise ValueError("waveform must be continuous or pulse.")
        copied = np.array(profile, dtype=float, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "profile", copied)

    def acceleration(self, time, *, frequency_scale=1.0, amplitude_scale=1.0):
        if not np.isfinite(time) or time < 0:
            raise ValueError("time must be finite and nonnegative.")
        if not np.isfinite(frequency_scale) or frequency_scale <= 0:
            raise ValueError("frequency_scale must be finite and positive.")
        if not np.isfinite(amplitude_scale) or amplitude_scale < 0:
            raise ValueError("amplitude_scale must be finite and nonnegative.")
        frequency = self.frequency * frequency_scale
        cycles = max(0.0, float(time)) * frequency
        if self.waveform == "continuous":
            fraction = min(cycles / self.ramp_cycles, 1.0)
            envelope = np.sin(np.pi * fraction / 2) ** 2
        elif self.waveform == "pulse":
            offset = (cycles - self.pulse_center_cycles) / self.pulse_width_cycles
            envelope = np.exp(-0.5 * offset**2)
        else:
            raise ValueError("waveform must be continuous or pulse.")
        omega = 2 * np.pi * frequency
        strength = 0.2 * omega**2 * self.amplitude * amplitude_scale
        return self.profile * strength * envelope * np.sin(omega * float(time))


@dataclass(frozen=True)
class SceneOverlay:
    """Visible geometry in SI coordinates, separate from the solver arrays."""

    label: str
    x: np.ndarray
    y: np.ndarray
    color: str = "#edc685"


@dataclass(frozen=True)
class WaveScene:
    """An explicit scene that can be passed to the time-domain solver."""

    name: str
    description: str
    grid: Grid
    refractive_index: np.ndarray
    damping: np.ndarray
    solid_mask: np.ndarray
    sources: tuple[WaveSource, ...]
    overlays: tuple[SceneOverlay, ...]
    wavelength: float
    c0: float = LIGHT_SPEED

    def __post_init__(self):
        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength must be finite and positive.")
        if not np.isfinite(self.c0) or self.c0 <= 0:
            raise ValueError("c0 must be finite and positive.")
        for name in ("refractive_index", "damping", "solid_mask"):
            array = np.asarray(getattr(self, name))
            if array.shape != self.grid.shape:
                raise ValueError(f"{name} must match the scene grid shape.")
            if name == "solid_mask":
                if array.dtype != bool:
                    raise ValueError("solid_mask must be boolean.")
            elif np.iscomplexobj(array) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite and real.")
            copied = np.array(array, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        if np.any(self.refractive_index <= 0) or np.any(self.damping < 0):
            raise ValueError("Index must be positive and damping nonnegative.")
        if not self.sources or any(
            source.profile.shape != self.grid.shape for source in self.sources
        ):
            raise ValueError("Provide source profiles matching the scene grid.")
        if self.maximum_frequency_scale < 1 - 1e-9:
            raise ValueError(
                "Scene sources need at least 12 points per local wavelength."
            )

    @property
    def frequency(self):
        return self.c0 / self.wavelength

    @cached_property
    def maximum_frequency_scale(self):
        """Largest source multiplier retaining 12 points per local wavelength."""
        dx, dy = self.grid.spacing
        maximum_frequency = self.c0 / (12 * np.max(self.refractive_index) * max(dx, dy))
        return float(
            maximum_frequency / max(source.frequency for source in self.sources)
        )

    def forcing(self, time, *, frequency_scale=1.0, amplitude_scale=1.0):
        if not np.isfinite(frequency_scale) or frequency_scale <= 0:
            raise ValueError("frequency_scale must be finite and positive.")
        if frequency_scale > self.maximum_frequency_scale * (1 + 1e-9):
            raise ValueError(
                "The selected frequency needs a finer grid: "
                "require 12 points per local wavelength."
            )
        if not np.isfinite(amplitude_scale) or amplitude_scale < 0:
            raise ValueError("amplitude_scale must be finite and nonnegative.")
        acceleration = np.zeros(self.grid.shape)
        for source in self.sources:
            acceleration += source.acceleration(
                time,
                frequency_scale=frequency_scale,
                amplitude_scale=amplitude_scale,
            )
        acceleration[self.solid_mask] = 0
        return acceleration

    def make_solver(self, *, courant=0.65):
        """Create a fresh state; scene construction never advances the solver."""
        from .time_domain import WaveEquation2D

        return WaveEquation2D(
            self.grid,
            self.refractive_index,
            self.damping,
            solid_mask=self.solid_mask,
            c0=self.c0,
            courant=courant,
        )


def wave_grid(wavelength, *, points_per_wavelength=18, width=64, height=32):
    """Rectangular mesh with widths expressed in vacuum wavelengths."""
    if not np.all(np.isfinite([wavelength, points_per_wavelength, width, height])):
        raise ValueError("Grid parameters must be finite.")
    if wavelength <= 0 or points_per_wavelength < 12 or width <= 0 or height <= 0:
        raise ValueError(
            "Use a positive wavelength and at least 12 vacuum samples per wavelength."
        )
    nx = round(width * points_per_wavelength)
    ny = round(height * points_per_wavelength)
    if min(nx, ny) < 3:
        raise ValueError("Grid needs at least three samples per axis.")
    spacing = wavelength / points_per_wavelength
    x = (np.arange(nx) - (nx - 1) / 2) * spacing
    y = (np.arange(ny) - (ny - 1) / 2) * spacing
    return Grid(*np.meshgrid(x, y))


def sponge_damping(grid, wavelength, *, thickness=4.0, strength=4.0, c0=LIGHT_SPEED):
    """Quadratic damping rate near every edge, in inverse seconds."""
    if not np.all(np.isfinite([wavelength, thickness, strength, c0])):
        raise ValueError("Sponge parameters must be finite.")
    if min(wavelength, thickness, c0) <= 0 or strength < 0:
        raise ValueError(
            "Sponge length and speed must be positive; strength must be nonnegative."
        )
    x, y = grid
    distance = np.minimum.reduce((x - x.min(), x.max() - x, y - y.min(), y.max() - y))
    fraction = np.clip(1 - distance / (thickness * wavelength), 0, 1)
    return strength * (c0 / wavelength) * fraction**2


def finite_line_source(
    grid, wavelength, *, x=-24.0, half_height=10.0, waveform="continuous"
):
    """Soft bidirectional source sheet; its left-going wave enters the sponge."""
    if (
        not np.all(np.isfinite([wavelength, x, half_height]))
        or wavelength <= 0
        or half_height <= 0
    ):
        raise ValueError(
            "Source wavelength and half-height must be finite and positive."
        )
    profile = np.exp(-0.5 * ((grid.x / wavelength - x) / 0.22) ** 2)
    profile *= np.exp(-((grid.y / (half_height * wavelength)) ** 8))
    return WaveSource(
        profile=profile, frequency=LIGHT_SPEED / wavelength, waveform=waveform
    )


def lens_scene(*, wavelength=532e-9, points_per_wavelength=18, waveform="continuous"):
    """A true refractive-index lens: intersecting circles, n = 1.5.

    Radius 18λ and center thickness 4λ give a clear diameter about 16.5λ.
    This is a scalar speed-index model; it is not a vector electromagnetic
    treatment of arbitrary polarization or material dispersion.
    """
    grid = wave_grid(wavelength, points_per_wavelength=points_per_wavelength)
    if points_per_wavelength < 18:
        raise ValueError(
            "The n=1.5 lens needs at least 18 vacuum samples per wavelength."
        )
    center = -8 * wavelength
    radius = 18 * wavelength
    thickness = 4 * wavelength
    displacement = radius - thickness / 2
    left_circle = (grid.x - center + displacement) ** 2 + grid.y**2 <= radius**2
    right_circle = (grid.x - center - displacement) ** 2 + grid.y**2 <= radius**2
    material = left_circle & right_circle
    refractive_index = np.where(material, 1.5, 1.0)

    half_height = np.sqrt(radius**2 - displacement**2)
    y = np.linspace(-half_height, half_height, 241)
    half_width = np.sqrt(radius**2 - y**2) - displacement
    outline_x = np.concatenate(
        (center - half_width, (center + half_width)[::-1], [center])
    )
    outline_y = np.concatenate((y, y[::-1], [y[0]]))
    overlays = (SceneOverlay("n = 1.5", outline_x, outline_y),)
    return WaveScene(
        name="lens",
        description="A dielectric biconvex lens focuses a driven scalar wave",
        grid=grid,
        refractive_index=refractive_index,
        damping=sponge_damping(grid, wavelength),
        solid_mask=np.zeros(grid.shape, dtype=bool),
        sources=(
            finite_line_source(grid, wavelength, half_height=7.0, waveform=waveform),
        ),
        overlays=overlays,
        wavelength=wavelength,
    )


def double_slit_scene(
    *, wavelength=532e-9, points_per_wavelength=18, waveform="continuous"
):
    """Two apertures in a reflecting Dirichlet wall, not an absorbing screen."""
    grid = wave_grid(wavelength, points_per_wavelength=points_per_wavelength)
    wall_position = -8 * wavelength
    wall = np.abs(grid.x - wall_position) <= 0.2 * wavelength
    openings = (np.abs(grid.y / wavelength - 4) < 1) | (
        np.abs(grid.y / wavelength + 4) < 1
    )
    solid = wall & ~openings
    overlay_parts = []
    for lower, upper in ((-16, -5), (-3, 3), (5, 16)):
        overlay_parts.append(
            SceneOverlay(
                "reflecting wall",
                np.array([wall_position, wall_position]),
                np.array([lower, upper]) * wavelength,
                color="#e8e6df",
            )
        )
    return WaveScene(
        name="double_slit",
        description="Two 2λ openings in a reflecting wall emit interfering waves",
        grid=grid,
        refractive_index=np.ones(grid.shape),
        damping=sponge_damping(grid, wavelength),
        solid_mask=solid,
        sources=(
            finite_line_source(grid, wavelength, half_height=11.0, waveform=waveform),
        ),
        overlays=tuple(overlay_parts),
        wavelength=wavelength,
    )
