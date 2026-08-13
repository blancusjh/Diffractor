"""An interface is a geometric locus separating two media.  Nothing more.

    Medium 1  —  Σ  —  Medium 2

The profile is pure geometry (:mod:`diffractor.geometry`); the media are pure
matter (:mod:`diffractor.optics.media`).  This class only binds them.  What
light DOES at the interface belongs elsewhere:

* local response coefficients (t_s, T_s)      → :mod:`diffractor.scattering`
* energy transport across it (ray tubes, 1/d) → :mod:`diffractor.propagation`
* path lengths, wavefronts, pupils            → :mod:`diffractor.analysis`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..geometry.descartes import DescartesOvoid
from .media import Medium

__all__ = ["Interface", "stigmatic_interface"]


@dataclass(frozen=True)
class Interface:
    """The locus Σ between ``medium1`` (incidence side) and ``medium2``.

    ``profile`` is a geometry object (e.g. :class:`DescartesOvoid`) exposing
    ``sample()`` / ``geometry()``; this class adds no behaviour to it.
    """

    profile: Any
    medium1: Medium
    medium2: Medium

    @property
    def n1(self) -> float:
        return self.medium1.n

    @property
    def n2(self) -> float:
        return self.medium2.n

    def sample(self, *args, **kwargs):
        """Delegate to the profile's geometric sampler (pure geometry out)."""
        return self.profile.sample(*args, **kwargs)


def stigmatic_interface(medium1: Medium, medium2: Medium,
                        zo: float, zi: float) -> Interface:
    """The interface whose profile is the Descartes ovoid imaging A(zo)→A'(zi).

    Stigmatism is a property of this (profile, media) pairing — the curve is
    built with the same n1, n2 it separates — not extra behaviour of the
    Interface class.
    """
    return Interface(
        profile=DescartesOvoid(medium1.n, medium2.n, zo, zi),
        medium1=medium1,
        medium2=medium2,
    )
