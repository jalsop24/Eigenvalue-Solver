# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 22:27:29 2021

@author: James
"""

import numpy as np
import matplotlib.pyplot as plt

def doubleDot(Vmax, a, b, x):
    
    V = Vmax/np.power(b, 4) * np.square( (x - 0.5*a)**2 - b**2 )
    
    return V
    
    


# Vmax = 1
# a = 0
# b = 25

# Xa = -40
# Xb = 40
# Nx = 100

# x = np.linspace(Xa, Xb, Nx)

# V = doubleDot(Vmax, a, b, x)


# plt.plot(x, V)
# plt.xlabel("x / au")
# plt.ylabel("V / au")
# plt.ylim(0, 2)











































