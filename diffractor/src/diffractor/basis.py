"""Bases of the plane, and the maps between them.

A :class:`~diffractor.space.Grid` samples the plane in some coordinate basis.
The basis is data, not behaviour: a name plus the two maps that carry its
coordinates to and from the canonical cartesian pair.  The built-in bases are
the module singletons :data:`CARTESIAN` and :data:`POLAR`; a custom basis is
the same object built by the caller, with its maps supplied explicitly.

Everything downstream that must be basis-agnostic (quadrature over arbitrary
node sets, the non-separable Fourier path) works through ``to_cartesian`` —
the plane is one space, the bases are only ways of addressing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Union

import numpy as np

__all__ = ["Basis", "CARTESIAN", "POLAR", "resolve_basis",
           "polar_to_cartesian", "cartesian_to_polar"]

_Pair = Tuple[np.ndarray, np.ndarray]


def polar_to_cartesian(r, theta) -> _Pair:
    """(r, θ) → (r cosθ, r sinθ)."""
    r = np.asarray(r, float)
    theta = np.asarray(theta, float)
    return r * np.cos(theta), r * np.sin(theta)


def cartesian_to_polar(x, y) -> _Pair:
    """(x, y) → (√(x²+y²), atan2(y, x)); atan2(0, 0) = 0, so the origin is safe."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return np.hypot(x, y), np.arctan2(y, x)


def _identity(u, v) -> _Pair:
    return np.asarray(u, float), np.asarray(v, float)


@dataclass(frozen=True, eq=False)
class Basis:
    """A coordinate basis of the plane: a name and its cartesian maps.

    ``to_cartesian(u, v)`` and ``from_cartesian(x, y)`` map coordinate arrays
    elementwise; both must be inverses of each other on the basis's domain.
    """

    name: str
    to_cartesian: Callable[[np.ndarray, np.ndarray], _Pair]
    from_cartesian: Callable[[np.ndarray, np.ndarray], _Pair]


#: The two built-in bases.  Their transformations live above in this module;
#: a Grid built from a name string resolves to one of these singletons.
CARTESIAN = Basis("cartesian", _identity, _identity)
POLAR = Basis("polar", polar_to_cartesian, cartesian_to_polar)

_BUILTIN = {"cartesian": CARTESIAN, "polar": POLAR}


def resolve_basis(basis: Union[str, Basis]) -> Basis:
    """Resolve a basis name or instance to a :class:`Basis`.

    Strings name a built-in basis; anything else must be a ``Basis`` carrying
    its own maps (that is what "custom" means — the basis is specified, not
    implied).
    """
    if isinstance(basis, Basis):
        return basis
    if isinstance(basis, str):
        try:
            return _BUILTIN[basis]
        except KeyError:
            raise ValueError(
                f"unknown basis {basis!r}; allowed: 'cartesian', 'polar', "
                f"or a Basis instance with explicit maps"
            ) from None
    raise TypeError(f"basis must be a str or Basis, got {type(basis).__name__}")
