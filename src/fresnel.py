import numpy as np 
array = np.array


from geometry import dir_A_to_P 




u = dir_A_to_P(A, Q) 


def cosines_from_u_and_N(u : array, u_ : array, N : array): 

    cos_i = -u.dot(N) 
    cos_t = -u_.dot(N)

    return cos_i, cos_t


# Fresnel Coefficients. 

def ts(n1, n2, cos_i, cos_t): 
    return 2*n1*cos_i/(n1*cos_i + n2*cos_t  

def tp(n1, n2, cos_i, cos_t): 
    return 2*n1*cos_i/(n2*cos_i + n1*cos_t)

# Fresnel Polarization basis.

def unit_s(u : Vec, N : Vec) -> Vec: 
    """Compute s polarization unit vector
    
    Parameters 
    u : Vec, unit vector incidence/transmision vector. 
    N : Vec, unit normal vector in a point Q of the diopter.

    """

    ss = np.cross(u, N)
    ss_norm = np.linalg.norm(ss) 

    hat_s = ss/ss_norm 
    
    return hat_s

    
def unit_p(s : Vec, u : Vec) -> Vec: 
    """Compute p polarization uint vector

    Parameters 
    u : Vec, unit vector incidence/transmision vector. 
    N : Vec, unit normal vector in a point Q of the diopter.

    """

    unit_p = np.cross(s, u)

    return unit_p


def fresnel_transmission_matrix(p, p_, s, N): 

    """
    Built the Fresnel Tranmission Matrix, 
    when this matrix multiplies the electric field in certain surface 
    Σ, it returns the transmited field in the same surface. 
    """


    s  = unit_s(u, N)
    p  = unit_p(s, u)
    p_ = unit_p_(s,u_) 

    T1 = array([p, s])
    T2 = array([p_, s]).T

    tp, ts = self.tp, self.ts

    T = array([[tp, 0],
               [0 , ts]])


    return T2@T@T1








