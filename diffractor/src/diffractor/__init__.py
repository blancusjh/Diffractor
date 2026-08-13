"""Diffractor v2 — scalar wave optics with physically correct interfaces.

The order of things:
    geometry/     pure shape          optics/       matter and its boundaries
    scattering/   interface response  propagation/  transport through space
    sources/      what emits          analysis/     what is measured

Validation contract: nothing in the physical core ships without a benchmark
against `groundtruth` (BOR-BEM ← exact scalar solutions).  See PLAN.md.
"""
from .optics import Medium, Interface, stigmatic_interface

__all__ = ["Medium", "Interface", "stigmatic_interface"]
