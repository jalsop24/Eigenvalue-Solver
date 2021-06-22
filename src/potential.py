# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 22:27:29 2021

@author: James
"""

import numpy as np
import matplotlib.pyplot as plt

from input.potential_data import pointEval

def doubleDot(Vmax, a, b, x):
    
    # V = Vmax/np.power(b, 4) * np.square( (x - 0.5*a)**2 - b**2 )
    
    
    V = a*np.square( x ) + Vmax * np.exp( -np.square( x/b ) )
    
    return V
    

def harmonic(m, w, x):
    return 0.5 * m * w**2 * np.square(x)


def harmonic2D(m, w, x, y):
    
    return 0.5 * m * w**2 * ( np.square(x) + np.square(y) )
    

def real2D(x, y):
    
    potential = np.zeros(shape=(len(y), len(x)))

    #potential = pointEval(x, y)
    for j in range(len(y)):
        val = pointEval(x, y[j])

        potential[:][j] = val

    return potential


def saw(x, A_saw, k_saw, w_saw, t_saw):
    return A_saw * np.cos(k_saw*x - w_saw*t_saw)




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











































