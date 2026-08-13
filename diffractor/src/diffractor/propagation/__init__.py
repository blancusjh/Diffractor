"""Field and energy transport through homogeneous regions.

exact/     — ASM, RS1, Hankel: solve Helmholtz with no further hypothesis
paraxial/  — Fresnel, gated to its validity regime
transport/ — general energy-conservation factors (ray tubes, n·cosθ flux)
"""
from .exact import hankel, asm_axisym, rs1_plane, fresnel_plane, spectral_budget
from .transport import ray_tube_amplitude, stigmatic_pupil

__all__ = ["hankel", "asm_axisym", "rs1_plane", "fresnel_plane",
           "spectral_budget", "ray_tube_amplitude", "stigmatic_pupil"]
