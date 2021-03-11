# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:10:25 2021

@author: James
"""



import numpy as np
import scipy.sparse as sp
from scipy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
from time import time

from src.makepath import makepath
from src.hamiltonian import generateH, generateHsp, genH2Psp
import src.potential as potential
from src.wavefunction import wavefunction
from src.fftSolver import solve as fftSolver


experimentName = "test"

Nx = 500

m = 1

Xa = -8
Xb = 8
dx = (Xb-Xa)/(Nx)

hbar = 1

clip = 50

w = 1

n = 0

N = 100

x = np.linspace(Xa, Xb, Nx, dtype=np.float64)


def FT(v, **kwargs):
    return fft(v, norm="ortho", **kwargs) 


def iFT(v, **kwargs):
    return ifft(v, norm="ortho", **kwargs) 

def coulombPot(index_diff):
    
    distance = abs(index_diff * dx)
    
    e0 = 0.05
    
    if distance < dx:
        return 1/(4*np.pi*e0*dx)
    else:
        return 1/(4*np.pi*e0*(distance))

V = potential.harmonic(m, w, x)

#V = potential.doubleDot(Vmax=0.1, a=0, b=15, x=x)

HK = generateHsp(dx, m, hbar, Nx)

# H = HK + sp.dia_matrix((V, [0]), shape=(Nx, Nx), dtype=np.float64)

H, V2 = genH2Psp(dx, m, hbar, Nx, Vi=V, Vc=coulombPot)



t0 = time()
eVals, eVecs = fftSolver(H, N, k=min(clip, Nx-1), sigma=0)
t1 = time()



dt = t1 - t0
print(dt)

zeroVec = eVecs[:,n]

eigOut = wavefunction(zeroVec, Nx, 2).normalise(dx)


# Plotting #

fig = plt.figure(dpi=400, figsize=(10, 7))
fig.suptitle(experimentName)

ax1 = fig.add_subplot( 1, 1, 1 ) #  , aspect=( len(eVals)*hbar*w/np.max(eVals).real ) 
# ax1.plot(x, eigOut.prob() )
ax1.plot(x, eigOut.prob1() )
ax1.plot(x, eigOut.prob2() )
# ax1.plot(hfft)
# ax1.plot(test)
ax1.plot(x, V)


ylim = 0.5
ax1.set_ylim(-ylim, ylim)
# ax1.set_xlim(Xa, Xb)


ax1.set_title(f"N = {n} Eigenfunction, Nx = {Nx}")
ax1.set_xlabel("x/au")
ax1.set_ylabel("$ |\Psi|^2 $")


# fig.savefig(outDir + experimentName + "/plot.png")



















