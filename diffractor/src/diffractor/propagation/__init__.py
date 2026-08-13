"""Field and energy transport through homogeneous regions.

exact/     — ASM, RS1, Hankel: solve Helmholtz with no further hypothesis
paraxial/  — Fresnel, gated to its validity regime
transport/ — general energy-conservation factors (ray tubes, n·cosθ flux)
"""
from .transport import ray_tube_amplitude, stigmatic_pupil
__all__ = ["ray_tube_amplitude", "stigmatic_pupil"]
