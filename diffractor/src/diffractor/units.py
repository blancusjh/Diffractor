"""Length units, metre base.

Six floats, nothing else: multiply on the way in (``wavelength=550*nm``) and
divide on the way out (``r / um`` for an axis label).  A units *library* was
rejected — the package is homogeneous SI, so carrying dimension checks through
every array operation buys nothing and costs everywhere.
"""

m: float = 1.0
cm: float = 1e-2 * m
mm: float = 1e-3 * m
um: float = 1e-6 * m
nm: float = 1e-9 * m

__all__ = ["m", "cm", "mm", "um", "nm"]
