# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 20:50:22 2021

@author: James
"""

import numpy as np
#from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
import scipy.sparse as sp
import scipy.linalg as LA
import scipy.sparse.linalg as SLA
from scipy.special import eval_hermite, factorial
import pyamg
import scipy.interpolate as lerp
import scipy.fft as fft

from src.wavefunction import wavefunction
from src.makepath import makepath
from src.hamiltonian import generateH, generateHsp, genH2Psp
import src.potential as potential
import src.gpuSolver as eigGPU
import src.fftSolver as fftSolver
import src.hamiltonian as hamiltonian

outDir = "./data/"

experimentName = "test"

m = 1 

Xa = -1700
Xb = 1700

Ni = 15 # Number of repetitions of the experiment to do in a single run

hbar = 1

w = 0.07

Nx = 200

Nr = 10 # Number of runs to carry out

clip = 50 # First "clip" eigenvalues will be found

maxNx = 1000

ncv = None

N = 100 # Number of k space points per dimension

k = Nr/np.log(maxNx - 11 - Nr)

start = 100
step = 20
NxArray = np.arange(start, start + Nr*step, step)

start = 5
step = 5
clipArray = np.arange(start, start + Nr*step, step)


start = 50
step = 5
N_Array = np.arange(start, start + Nr*step, step)

start = 51
step = 1
ncvArray = np.arange(start, start + Nr*step, step)

start = 0
step = 1
nArray = np.arange(start, start + Nr*step, step)

eVals, eVecs = None, None 

n = 49

a = m*w/hbar

k_saw = 0.012
w_saw = 0.1
A_saw = 0.001
t_saw = 0

def runExperiment(hbar, m, w, Xa, Xb, Ni, Nx, Ny, ncv, n, clip, N):
    
    x = np.linspace(Xa,Xb,Nx)
    dx = (Xb-Xa)/(Nx-1)

    # Yval = 430
    Ya = -430
    Yb = 430
    dy = (Yb-Ya)/(Ny-1)
    
    #V = potential.harmonic(m, w, x)
    
    def coulombPot(index_diff):
    
        distance = abs(index_diff * dx)
    
        e0 = 0.05
    
        if distance < dx:
            return 1/(4*np.pi*e0*dx)
        else:
            return 1/(4*np.pi*e0*(distance))
    
    y = np.sqrt(a) * x

    eigTheory = np.complex64( np.power(2.0**n * factorial(n), -0.5) * np.power(a/np.pi, 0.25) * eval_hermite(n, y) * np.exp(-0.5*y**2 ) ) 

    theoryWavefunc = wavefunction(eigTheory, Nx).normalise(dx)
    
    # V = potential.doubleDot(Vmax=0.25, a=0.5*m*w**2, b=4, x=x)


    V = (potential.real2D(x, y) + potential.saw(x, A_saw, k_saw, w_saw, t_saw)).ravel()
    
    # V = potential.harmonic(m, w, x)

    #HK = generateH(dx, m, hbar, Nx)

    # H = HK + np.diag(V)
    
    # HKsp = generateHsp(dx, m, hbar, Nx)
    
    # H = generateHsp(dx, m, hbar, Nx, V)
    
    # H = genH2Psp(dx, m, hbar, Nx, Vi = V, Vc = coulombPot)
    
    H = hamiltonian.genH2Dsp(dx, dy, Nx, Ny, m, hbar, Vi = V)
    
    # Hinv = SLA.inv(H.tocsc())
    
    # M = SLA.aslinearoperator(Hinv)
    
    # M = pyamg.smoothed_aggregation_solver(H.tocsr()).aspreconditioner()
    
    # eVals0, eVecs0 = SLA.eigsh(H, k = min(clip, Nx - 2), sigma=0, return_eigenvectors=True)
    
    # Ns = min(Nx, 1000)
    # dxs = (Xb-Xa)/(Ns-1)
    # xs = np.linspace(Xa, Xb, Ns, dtype=np.float64)

    # Vs = potential.harmonic(m, w, xs)
    # Hs = generateHsp(dxs, m, hbar, Ns, Vs, dtype=np.float64)

    # eVals0, eVecs0 = SLA.eigsh( Hs, k=min(clip, Nx-2), sigma=0, return_eigenvectors=True)

    # eVecs0 = lerp.interp1d(xs, eVecs0, axis=0, kind='quadratic')(x) /np.sqrt(Nx/Ns)
    
    # # eVecs0 = np.random.rand(H.shape[0], clip)
    
    eVecsR, fidelity = None, None
    

    
    
    times = np.zeros((Ni))
    
    for run_number in range(Ni):
        
        startT = time.time()
        
        # eVals, eVecs = LA.eigh( H.toarray(), eigvals_only=False, subset_by_index=[0, min(clip, Nx) - 1] )
        eVals, eVecs = SLA.eigsh(H, k =min(clip, Nx-2), sigma=0, return_eigenvectors=True, ncv=ncv)
        # eVals, eVecs = eigGPU.solve_sp(H, k=min(clip, Nx-1), mu0=eVals, x0=eVecs, dtype=np.float64)
        
        # eValsR, eVecsR = SLA.eigsh(H, k =min(clip, Nx-2), sigma=0, return_eigenvectors=True, ncv=ncv)
        # eVals, eVecs, dt = fftSolver.solve(H, N, p=1, eigvals_only=False, subset_by_index=[0, min(clip, Nx-1)] )
        
        # eVals, eVecs, dt = fftSolver.solve_gpu(H, N, p=1, lib="cusolver", jobvl="N", imag="T")
        
        # eVals, eVecs = SLA.lobpcg( H, eVecs0, M=M, largest=False, tol=None)
        endT= time.time()
        
        times[run_number] = endT - startT
        
    
    
    idt = times.mean()
    sd = times.std() 
    
    # fidelity = theoryWavefunc.inner( wavefunction(eVecs[:, n], Nx).normalise(dx), dx)
    # fidelity = wavefunction(eVecsR[:,n], Nx).normalise(dx).inner( wavefunction(eVecs[:, n], Nx).normalise(dx), dx)
    
    nnz = H.nnz
    
    return idt, sd, eVals, eVecs, fidelity, nnz

tdata = []
sd_data = []
xdata = []
# fdata = []
ndata = []

t0 = time.time()

for i in range(0, Nr):
    
    Nx = NxArray[i] # int( 10 + n + np.exp(i/k) ) # 
    Ny = Nx
    
    # Nx = fft.next_fast_len(Nx)
    
    # n = nArray[i]
    
    # clip = clipArray[i]
    
    # N = N_Array[i]
    
    # ncv = ncvArray[i]
    
    idt, sd, eVals, eVecs, fidelity, nnz = runExperiment(hbar, m, w, Xa, Xb, Ni, Nx, Ny, ncv, n, clip, N)
    
    dt = idt*Ni
    
    tdata.append(idt)
    xdata.append(Nx)
    sd_data.append(sd)
    # fdata.append(fidelity)
    ndata.append(nnz)
    
    # print(f"Nx: {Nx}, dt: {dt}")
    print(f"Run Number: {i+1}\n Nx: {Nx}, Clip: {clip}, dt: {dt}, idt: {idt}, F: {fidelity}")
    
dt = time.time() - t0



print(f"{Nr} experiments completed.")
print(f"Total time: {dt}")

makepath(experimentName)


xdata = np.array(xdata, dtype=np.int)
tdata = np.array(tdata)


with open(outDir + experimentName + "/params.txt", "w+") as file:
    file.write( f"Xa = {Xa},\nXb = {Xb},\nNi = {Ni},\ndt = {dt},\nw = {w},\nclip = {clip}" ) 

np.savetxt(outDir + experimentName + "/x_values.txt", xdata)
np.savetxt(outDir + experimentName + "/dt_values.txt", tdata)
np.savetxt(outDir + experimentName + "/sd_values.txt", sd_data)
# np.savetxt(outDir + experimentName + "/f_values.txt", fdata)
np.savetxt(outDir + experimentName + "/nnz_values.txt", ndata)


plt.plot(xdata, tdata )

# plt.plot(xdata, np.log10( np.abs( 1 - np.array(fdata) ) ) )

plt.xlabel("$x$")
plt.ylabel("Time to solve / s")




























