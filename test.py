from dataclasses import dataclass 

import numpy as np


@dataclass 
class C: 
    a : float 
    b : float 
    m : str 




ii = np.array([1, 0, 0])
jj = np.array([0, 1, 0])
kk = np.array([0, 0, 1])


print(np.array([ii, jj]).T)
