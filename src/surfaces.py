import numpy as np 
from dataclasses import dataclass
from typing import Literal

from geometry import rr_axisymmetric_surface



@dataclass(frozen=True)
class CartesianSurface:
    """
    Surface parameters: (G, O, T, S)

    Profile functions:

        z(rho) = ((O + T rho^2) rho^2) /
                 (1 + S rho^2 + sqrt(1 + (2S - O^2 G) rho^2))

        r(rho) = ± sqrt(rho^2 - z(rho)^2)
    """
    no : float 
    ni : float 
    z0 : float 
    zi : float 


    def GOTS(no, ni, zo, zi): 

        G = (ni**2 - no**2)**2/(ni*no*(ni*zi - no*zo)*(ni*zo - no*zi))
        O = (ni*zo - no*zi)/(zi*zo*(ni - no))
        T = (ni - no)*(no + ni)**2/(4*ni*no*zi*zo(ni*zi - no*zo))
        S = (ni + no)*(ni**2*zi - no**2*zo)/(2*ni*no*zi*zo(ni*zi - no*zo))
        return G, O, T, S

    def z(self, rho: float) -> float:

        rho2 = rho**2
        G, O, T, S = GOTS(self.no, self.ni, self.zo, self.zi)

        return ((O + T * rho2) * rho2) / (
            1.0
            + S * rho2
            + np.sqrt(1.0 + (2.0 * S - O**2 * G) * rho2)
        )

    def r(self, rho: float) -> float:
        z_val = self.z(rho)
        return np.sqrt(rho**2 - z_val**2)


    def RR(self, rho: float, phi: float): 
        return rr_axisymmetric_surface(self.z, self.r, rho, phi) 


        







