# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 20:50:22 2021

@author: James
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as LA
import scipy.sparse.linalg as SLA
import matplotlib.pyplot as plt
import time
from scipy.special import eval_hermite, factorial
from src.makepath import makepath
from src.hamiltonian import generateH, generateHsp, genH2Psp
import src.potential as potential
import src.gpuSolver as eigGPU
from src.wavefunction import wavefunction
from src.fftSolver import solve as fftSolver

outDir = "./data/"

experimentName = "test"

Nx = 1000

m = 1

Xa = -50
Xb = 50
dx = (Xb-Xa)/(Nx-1)

Ni = 10

hbar = 1

clip = 10

n = 2

N = 50

w = 0.07

x = np.linspace(Xa,Xb,Nx, dtype=np.float64)

eVals, eVecs = None, None
    
def coulombPot(index_diff):
    
    distance = abs(index_diff * dx)
    
    e0 = 0.05
    
    if distance < dx:
        return 1/(4*np.pi*e0*dx)
    else:
        return 1/(4*np.pi*e0*(distance))
    

def inner(vec1, vec2, dx):
    return np.abs( np.inner(vec1.conj(), vec2) )*dx

# h2p = genH2P(dx, m, hbar, Nx)


t0 = time.time()

V = potential.harmonic(m, w, x)

# V = potential.doubleDot(Vmax=0.1, a=0, b=15, x=x)


# HK = generateH(dx, m, hbar, Nx)

# H = HK + np.diag(V)


t1 = time.time()

H = generateHsp(dx, m, hbar, Nx, V, dtype=np.complex64)

# H = genH2Psp(dx, m, hbar, Nx, Vi = V, Vc = coulombPot)


# eVals, eVecs = LA.eigh( H, eigvals_only=False, subset_by_index=[0, min(clip, Nx) - 1] )

# Hcsr = sp.csr_matrix(H)

#eVals, eVecs = solve(H.toarray())
eVals0, eVecs0 = SLA.eigsh( H, k=min(clip, Nx-2), sigma=0, return_eigenvectors=True)

t2 = time.time()

times = np.zeros((Ni, 6))

for i in range(Ni):
    # print(i)
    # eVals, eVecs = LA.eigh( H, eigvals_only=False, subset_by_index=[0, min(clip, Nx) - 1] )
    # eVals, eVecs = np.linalg.eigh(H.toarray())
    # eVals, eVecs = SLA.eigsh( H.tocsr() , k=min(clip, Nx-2), sigma=0, return_eigenvectors=True)
    # eVals = SLA.eigsh( H.tocsr(), k=min(clip, Nx-2), sigma=0, return_eigenvectors=False)
    # eVals, eVecs = eigGPU.solve_sp(H, k=min(clip, Nx-2), mu0=eVals, x0=eVecs, dtype=np.float64) # Returns zeros
    # eVals, eVecs = eigGPU.solve(H.toarray()) # CUSOLVER_STATUS_EXECUTION_FAILED
    # eVals, eVecs, dt = fftSolver(H, N, k=min(clip, Nx-1), sigma=0)
    eVals, eVecs, dt = fftSolver(H, N, p=1, eigvals_only=False, subset_by_index=[0, min(clip, Nx)-1] )
    times[i] = dt



times2 = np.mean(times, 0)

print(times2/np.sum(times2))


t3 = time.time()

dt = t3 - t0
idt = (t3-t2)/Ni

dt1 = t1 - t0
dt2 = t2 - t1

n = min(n, len(eVecs[0])-1)

print(f"{Ni} iterations completed.")
print(f"Total time: {dt}")
print(f"Average time per iteration: {idt}")
print(f"dt1 = {dt1},\ndt2 = {dt2}")
print(f"n = {n}")


makepath(experimentName)

with open(outDir + experimentName + "/params.txt", "w+") as file:
    file.write( f"Nx = {Nx},\nXa = {Xa},\nXb = {Xb},\nNi = {Ni},\nidt = {idt},\ndt = {dt},\nw = {w}" ) 
    
np.savetxt(outDir + experimentName + "/values.txt", eVals )
np.savetxt(outDir + experimentName + "/vectors.txt", eVecs )



fig = plt.figure(dpi=400, figsize=(16, 7))

fig.suptitle(experimentName)

nArray = np.arange(len(eVals))

ax1 = fig.add_subplot( 1, 2, 1 ) #  , aspect=( len(eVals)*hbar*w/np.max(eVals).real ) 
ax1.plot( eVals.real/(hbar*w) )
# ax1.plot( (nArray + 0.5) )
ax1.set_xlabel("n")
ax1.set_ylabel("Energy/ $\hbar \omega$")
ax1.set_title("Energy Eigenvalues")

a = m*w/hbar
y = np.sqrt(a) * x
z = np.sqrt(0.5*a) * x

eigTheory = np.complex64( np.power(2**n * factorial(n), -0.5) * np.power(a/np.pi, 0.25) * eval_hermite(n, y) * np.exp(-0.5*y**2 ) )

eigOut = eVecs[:,n] # eVecs 

# print( norm(eig3, dx) )
# print( norm(eigOut, dx))

eigOut = wavefunction(eigOut, Nx).normalise(dx)
eigTheory = wavefunction(eigTheory, Nx).normalise(dx)

print( inner(eigOut.Data, eigTheory.Data, dx) ) 

# print( norm(eigOut, dx) )

ax2 = fig.add_subplot(1, 2, 2, aspect=100)
# ax2.plot(x, eigOut.prob1() )
# ax2.plot(x, eigOut.prob2() )
# ax2.plot(x, eigOut.Data.real)
ax2.plot(x, eigOut.prob())
ax2.plot(x, V)
# ax2.plot(x, eigTheory.Data.real )
ylim = 0.5
ax2.set_ylim(-0.05, ylim)
ax2.set_xlim(Xa, Xb)
ax2.set_title(f"N = {n} Eigenfunction, Nx = {Nx}")
ax2.set_xlabel("x/au")
ax2.set_ylabel("$ |\Psi|^2 $")

fig.savefig(outDir + experimentName + "/plot.png")































