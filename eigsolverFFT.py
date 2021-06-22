# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:10:25 2021

@author: James
"""



import numpy as np
import scipy
import scipy.sparse as sp
from scipy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
from time import time

from src.makepath import makepath
from src.hamiltonian import generateH, generateHsp, genH2Psp
import src.potential as potential
from src.wavefunction import wavefunction
import src.fftSolver as fftSolver
from src.fftSolver import transformH 



experimentName = "test"

Nx = 200

m = 1

Xa = -100
Xb = 100
dx = (Xb-Xa)/(Nx)

hbar = 1

clip = 20

w = 0.02

n = 0

N = Nx

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


def inner(vec1, vec2, dx):
    result = 0
    
    for i in range(len(vec1)):
        
        result += vec1[i].conj() * vec2[i] * dx
        
    return np.abs(result)


V = potential.harmonic(m, w, x)

# V = potential.doubleDot(Vmax=0.1, a=0, b=30, x=x)

H = generateHsp(dx, m, hbar, Nx, V, dtype=np.complex64)

# H = genH2Psp(dx, m, hbar, Nx, Vi=V) # Vc = coulombPot



Hk, times = transformH(H, N)

HK = Hk.copy()

t0 = time()
# eVals, eVecs, dts = fftSolver(H, N, p=2, eigvals_only=False, subset_by_index=[0, min(clip, Nx-1)] )
eVals, eVecs = sp.linalg.eigsh( H.tocsr() , k=min(clip, Nx-2), sigma=0, return_eigenvectors=True)
eValsK, eVecsK = scipy.linalg.eigh(Hk, eigvals_only=False, subset_by_index=[0, min(clip, Nx)-1]  )
t1 = time()




dt = t1 - t0
print(dt)

zeroVec = eVecs[:,n]
zeroVecK = eVecsK[:,n]



# zeroMat = zeroVec.reshape(Nx, Nx)

eigOut2 = FT(zeroVec) # should be the same as zeroVecK

eigOut3 = iFT(eigOut2)

eigOut = wavefunction( zeroVec, Nx, 1).normalise(dx)
eigOut2 = wavefunction(fftshift(eigOut2), Nx, 1).normalise(dx)
eigOut3 = wavefunction(eigOut3, Nx, 1).normalise(dx)


eigK = wavefunction(zeroVecK, Nx).normalise(dx)


diff = eigOut2.Data / eigK.Data

k = 2*np.pi*1j/2


#zeroVecK = np.roll(zeroVecK, 100 )

eigOutFFT = iFT(fftshift( zeroVecK) ) 
eigOutFFT = wavefunction(eigOutFFT, Nx).normalise(dx)

print( inner(eigOutFFT.Data, eigOut.Data, dx) ) 

arg1 = np.angle(eigK.Data)
arg2 = np.angle(eigOut2.Data)


# ftreal = FT(eigOutFFT.Data.real) 




# Plotting #

fig = plt.figure(dpi=400, figsize=(10, 7))
fig.suptitle(experimentName)

ax1 = fig.add_subplot( 1, 1, 1 ) #  , aspect=( len(eVals)*hbar*w/np.max(eVals).real ) 
# ax1.plot(x, eigOut.Data.real )
# ax1.plot(x, eigOut.Data.imag )
ax1.plot(x, eigOutFFT.Data.real )
ax1.plot(x, eigOutFFT.Data.imag )
# ax1.plot(x, eigK.Data.real )
# ax1.plot(x, eigK.Data.imag )
# ax1.plot(x, eigOut2.Data.real )
# ax1.plot(x, eigOut2.Data.imag)
# ax1.plot(x, np.abs(eigK.Data - eigOut2.Data) )
# ax1.plot(x, eigOut3.Data )
# ax1.plot(x, eigOut.prob2() )
# ax1.plot(hfft)
# ax1.plot(test)
# ax1.plot(x, V)

# ax1.plot(ftreal)

ylim = 0.5
a = 1
ax1.set_ylim(-ylim, ylim)
# ax1.set_xlim(Xa/a, Xb/a)


ax1.set_title(f"N = {n} Eigenfunction, Nx = {Nx}")
ax1.set_xlabel("x/au")
ax1.set_ylabel("$ |\Psi|^2 $")


# fig.savefig(outDir + experimentName + "/plot.png")



















