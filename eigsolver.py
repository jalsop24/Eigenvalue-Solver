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
import pylops
import pyamg
from scipy.special import eval_hermite, factorial
import scipy.interpolate as lerp

from src.makepath import makepath
import src.hamiltonian as hamiltonian
import src.potential as potential
import src.gpuSolver as eigGPU
from src.wavefunction import wavefunction
import src.fftSolver as fftSolver

outDir = "./data/"

experimentName = "test"

Nx = 1_000
Xa = -45
Xb = 45
dx = (Xb-Xa)/(Nx-1)

Ny = 100
# Yval = 430
Ya = -430
Yb = 430
dy = (Yb-Ya)/(Ny-1)

Ni = 50

clip = 50

n = 5

N = 100

tol = 0
maxIt = 10

hbar = 1
m = 1
w = 0.07

k_saw = 0.012
w_saw = 0.1
A_saw = 0.01
t_saw = 0

x = np.linspace(Xa,Xb,Nx, dtype=np.float64)

y = np.linspace(Ya,Yb,Ny, dtype=np.float64)

eVals, eVecs = None, None
    
def coulombPot(index_diff):
    
    distance = abs(index_diff * dx)
    
    e0 = 0.01
    
    if distance < dx:
        return 1/(4*np.pi*e0*dx)
    else:
        return 1/(4*np.pi*e0*(distance))
    

def inner(vec1, vec2, dx):
    return np.abs( np.inner(vec1.conj(), vec2) )*dx

# h2p = hamiltonian.genH2P(dx, m, hbar, Nx)




V = potential.harmonic(m, w, x)

# V = potential.doubleDot(Vmax=0.25, a=0.5*m*w**2, b=4, x=x)

# Vsaw = potential.saw(x, A_saw, k_saw, w_saw, t_saw)

# V = (potential.real2D(x, y) + Vsaw).ravel()

# HK = hamiltonian.generateH(dx, m, hbar, Nx)

# H = HK + np.diag(V)




H = hamiltonian.generateHsp(dx, m, hbar, Nx, V, dtype=np.float64).tocsc()

# D, E = hamiltonian.generateHtd(dx, m, hbar, Nx, V, dtype=np.float64)

# H = hamiltonian.getLOP(dx, m, hbar, Nx, V)

# H = hamiltonian.genH2Psp(dx, m, hbar, Nx, Vi = V, Vc = None, dtype=np.float64)

# H = hamiltonian.genH2Dsp(dx, dy, Nx, Ny, m, hbar, Vi = V)

# H = hamiltonian.generateHLOP(dx, m, hbar, Nx, V, dtype=np.complex64)

# Hinv = SLA.inv(hamiltonian.generateHsp(dx, m, hbar, Nx, V, dtype=np.float64).tocsc())

# dense = Hinv.toarray()

# Hsp = H.tosparse().toarray()

# Hsp2 = H2.toarray()

# M = pyamg.smoothed_aggregation_solver(H.tocsr()).aspreconditioner()
M = pyamg.smoothed_aggregation_solver(H).aspreconditioner()
# M = sp.eyes(H.shape[0])

# M = SLA.inv(H)

# eVals, eVecs = LA.eigh( H, eigvals_only=False, subset_by_index=[0, min(clip, Nx) - 1] )

# Hcsr = sp.csr_matrix(H)

# eVals, eVecs = solve(H.toarray())

# eVals0, eVecs0 = SLA.eigsh( H, k=min(clip, Nx-2), sigma=0, return_eigenvectors=True)


#print()

t0 = time.time()

Ns = 900
dxs = (Xb-Xa)/(Ns-1)
xs = np.linspace(Xa, Xb, Ns, dtype=np.float64)

Vs = potential.harmonic(m, w, xs)
Hs = hamiltonian.generateHsp(dxs, m, hbar, Ns, Vs, dtype=np.float64)

eVals0, eVecs0 = SLA.eigsh( Hs, k=min(clip, Nx-2), sigma=0, return_eigenvectors=True)

# # plt.plot(xs, eVecs0[:,0])

# eVecs0 = lerp.interp1d(xs, eVecs0, axis=0, kind='quadratic')(x) /np.sqrt(Nx/Ns)

t1 = time.time()

# eVecL = eVecs0[:,10].copy()

# print(inner(eVecL, eVecL, 1))

# plt.plot(x, eVecs0[:,0])

eVecs0 = np.random.rand(H.shape[0], clip)

t2 = time.time()

times = np.zeros((Ni))

for i in range(Ni):
    # print(i)
    
    startT = time.time()
    
    # eVals, eVecs = LA.eigh( H.toarray(), eigvals_only=False )
    # eVals, eVecs = np.linalg.eigh(H.toarray())
    
    # eVals, eVecs = LA.eigh_tridiagonal(D, E, select="i", select_range=[0, min(clip, Nx)-1])
    
    eVals, eVecs = SLA.eigsh( H , k=min(clip, Nx-2), tol=tol, maxiter=maxIt, sigma=0, return_eigenvectors=True, OPinv=None)  
    # eVals, eVecs = SLA.eigsh( H , k=min(clip, Nx-2), tol=tol, maxiter=maxIt, sigma=0, return_eigenvectors=True, ncv=None, v0=eVecL)    
    
    # eVals, eVecs = eigGPU.solve_sp(H, k=min(clip, Nx-2), mu0=eVals, x0=eVecs, dtype=np.float64) # Returns zeros
    # eVals, eVecs = eigGPU.solve(H.toarray()) # CUSOLVER_STATUS_EXECUTION_FAILED
    
    
    # eVals, eVecs, dt = fftSolver.solve(H, N, p=2, eigvals_only=False, subset_by_index=[0, min(clip, Nx)-1] )
    
    # eVals, eVecs, dt = fftSolver.solve_gpu(H, N, p=1, lib="cusolver", jobvl="N", imag="T" )
    
    # eVals, eVecs = SLA.lobpcg( H, eVecs0, largest=False, tol=1e-3)
    # eVals, eVecs = SLA.lobpcg( H, eVecs0, M=SLA.aslinearoperator(Hinv), largest=False, tol=1e-4, verbosityLevel=0, maxiter=10)
    # eVals, eVecs = SLA.lobpcg( H, eVecs0, M=M, largest=False, tol=1e-4, verbosityLevel=0, maxiter=10)
    
    # eVals = H.eigs(neigs=50, symmetric=True)
    
    
    dt = time.time() - startT
    times[i] = dt


sd = times.std()
mean = times.mean()

print("sd:", sd, sd/mean)
print("mean", mean)

# times2 = np.mean(times, 0)

# print(times2/np.sum(times2))

# print(H.nnz)

t3 = time.time()

dt = t3 - t0
idt = (t3-t2)/Ni

dt1 = t1 - t0
dt2 = t2 - t1

# print("inner", inner(eVecs[:,10], eVecL, 1))

n = min(n, len(eVecs[0])-1)

print(f"{Ni} iterations completed.")
print(f"Total time: {dt}")
print(f"Average time per iteration: {idt}")
print(f"dt1 = {dt1},\ndt2 = {dt2}")
print(f"n = {n}")


makepath(experimentName)

with open(outDir + experimentName + "/params.txt", "w+") as file:
    file.write( f"Nx = {Nx},\nXa = {Xa},\nXb = {Xb},\nNi = {Ni},\nidt = {idt},\ndt = {dt},\nw = {w}" ) 
    
# np.savetxt(outDir + experimentName + "/values.txt", eVals )
# np.savetxt(outDir + experimentName + "/vectors.txt", eVecs )


a = m*w/hbar
y = np.sqrt(a) * x

eigTheory = np.complex64( np.power(2**n * factorial(n), -0.5) * np.power(a/np.pi, 0.25) * eval_hermite(n, y) * np.exp(-0.5*y**2 ) ) 

eigOut = eVecs[:,n] # eVecs 

# # print( norm(eig3, dx) )
# # print( norm(eigOut, dx))

eigOut = wavefunction(eigOut, Nx).normalise(dx)
eigTheory = wavefunction(eigTheory, Nx).normalise(dx)

# print( norm(eigOut, dx) )

# print("F:", eigOut.inner(eigTheory, dx))

nArray = np.arange(len(eVals))



fig = plt.figure(dpi=400, figsize=(24, 7))

fig.suptitle(experimentName)

ax1 = fig.add_subplot( 1, 3, 1 ) #  , aspect=( len(eVals)*hbar*w/np.max(eVals).real ) 
ax1.plot( np.abs(eVals.real)/(hbar*w) )
# ax1.plot( (nArray + 0.5) )
ax1.set_xlabel("n")
ax1.set_ylabel("Energy/ $\hbar \omega$")
ax1.set_title("Energy Eigenvalues")



ax2 = fig.add_subplot(1, 3, 2)
# ax2.plot(x, eigOut.prob1() )
# ax2.plot(x, eigOut.prob2() )
# ax2.plot(x, eigOut.Data.real)
ax2.plot(x, eigOut.prob())
ax2.plot(x, V)
# ax2.plot(x, eigTheory.Data.real )
ylim = 0.5
ax2.set_ylim(-0.05, ylim)
# ax2.contourf(x, y, np.abs(eVecs[:,n].reshape(Ny, Nx))**2, 20 )

# ax2.set_xlim(Xa, Xb)
ax2.set_title(f"N = {n} Eigenfunction, Nx = {Nx}, $ |\Psi|^2 $")
# ax2.set_xlabel("k")
ax2.set_ylabel("$ |\Psi|^2 $")
# ax2.set_ylabel("y/au")




# ax3 = fig.add_subplot(1, 3, 3)
# ax3.set_title("Potential Landscape")
# ax3.contourf(x, y, V.reshape(Ny, Nx), 40, cmap="inferno")
# ax3.set_xlabel("x/au")
# ax3.set_ylabel("y/au")

fig.savefig(outDir + experimentName + "/plot.png")































