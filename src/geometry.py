import numpy as np 
from dataclasses import dataclass 
from typing import Callable, Optional 

Vec = np.ndarray 
norm = np.linalg.norm 


cos = np.cos 
sin = np.sin


ii = array([1, 0, 0]) 
jj = array([0, 1, 0])
kk = array([0, 0 ,1])



# Math / Geometry Utilities. 

def diff_c(f : Callable[float], 
                       x : float, 
                       h = 1e-6) -> Vec: 
    return (f(x + h) - f(x - h, phi))/(2.0*h)



def rr_axisymmetric_surface(z: Callable[[float], float], r: Callable[[float], float], tao : float, phi : float) -> Vec:

    RR = r(tao)*cos(phi)*ii + r(tao)*sin(phi)*jj + z(tao)**kk

    return RR


def dir_A_to_P(A : Vec, P : Vec) 
    AP = norm(P - A) 

    return (P-A)/AP


def tangent_vectors(RR : Callable[[float], float], u0, v0) -> Vec: 
    """Compute Tanget vectors of a surface given in parametric form"""

    RR_in_v0 =  lambda u : RR_l(u, v0)
    RR_in_u0 = lambda v : RR_l(u0, v) 

    RR_u = diff_c(RR_in_u0) 
    RR_v = diff_c(RR_in_v0) 

    return RR_u, RR_v


def unit_normal(RR : Callable[float, float], u0, v0) -> Vec: 
    "Compute Unit Vector of surface given in parametric form" 

    RR_u, RR_v = tangent_vectors(RR, u0, v0) 

    N_ = np.cross(RR_u, RR_v) 
    N_norm = np.linalg.norm(N_) 

    N = N_/N_norm 

    return N



import numpy as np

def spherical_basis(R, theta, phi):
    sin = np.sin
    cos = np.cos

    ii = np.array([1.0, 0.0, 0.0])
    jj = np.array([0.0, 1.0, 0.0])
    kk = np.array([0.0, 0.0, 1.0])

    unit_R = (
        sin(theta) * cos(phi) * ii
        + sin(theta) * sin(phi) * jj
        + cos(theta) * kk
    )

    unit_theta = (
        cos(theta) * cos(phi) * ii
        + cos(theta) * sin(phi) * jj
        - sin(theta) * kk
    )

    unit_phi = (
        -sin(phi) * ii
        + cos(phi) * jj
    )

    position = R * unit_R

    return position, unit_R, unit_theta, unit_phi




