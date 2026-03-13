import numpy as np 
from surfaces import CartesianSurface
from geometry import *


norm = np.linalg.norm 
exp = np.exp




array = np.array


def spherical_wave(k, rr): 

    r = norm(rr)
    return exp(1.0j*k*r)/r




class Simple_Stigmatic_System: 

    def __init__(no : float, ni : float, z0 : Vec, zi_ : Vec_, alpha : float, λ = 533*1e-6)  
    
    """
    Stigmatic System formed by a diopter, ergo the form of the diopter 
    is a Cartesian surface. 
    This Simple Stigmatic System is determined by the two focus, and 
    the angle of aperture of the source.
    """

    self.z0 = z0 
    self.zi = zi 

    A  = array([0, 0, z0]) 
    A_ = array([0, 0, zi])
    self.A  = A
    self.A_ = A_


    self.Σ = CartesianSurface(no, ni, zo, zi) 
    self.k = no*2*π/λ 
    self.k_ = ni*k

    
    def u(self, Q):
        A = self.A 

        return (Q-A)/norm(Q-A)

    def u_(self, Q): 
        A_ = self.A_ 

        return (Q-A_)/norm(Q-A_)

    def N(self, Q):

        RR = self.Σ.RR(rho, phi) 
        return unit_normal(RR, rho, phi) 

    def E0(rho, phi, Eθ, Eφ): 

        Q = Q(rho, phi) 

        Sph = spherical_wave(self.k, Q - A) 

        _, e_theta, e_phi = spherical_basis(theta, phi) 

        return Eθ*e_theta + Eφ*e_phi

        
        
 













    




