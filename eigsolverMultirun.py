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
from src.makepath import makepath
from src.hamiltonian import generateH, generateHsp
import src.potential as potential
import src.gpuSolver as eigGPU
from src.fftSolver import solve as fftSolver


outDir = "./data/"

experimentName = "test"

m = 1 

Xa = -50
Xb = 50

Ni = 5 # Number of repetitions of the experiment to do in a single run

hbar = 1

w = 0.07

Nx = 1000

Nr = 60 # Number of runs to carry out

clip = 50 # First "clip" eigenvalues will be found

maxNx = 1000

N = 100

k = Nr/np.log(maxNx - 11 - Nr)

start = 500
step = 1500
NxArray = np.arange(start, start + Nr*step, step)

start = 5
step = 5
clipArray = np.arange(start, start + Nr*step, step)


start = 50
step = 2
N_Array = np.arange(start, start + Nr*step, step)



eVals, eVecs = None, None 

def runExperiment(hbar, m, w, Xa, Xb, Ni, Nx, clip, N):
    
    x = np.linspace(Xa,Xb,Nx)
    dx = (Xb-Xa)/(Nx-1)
    
    V = potential.harmonic(m, w, x)
    
    # V = potential.doubleDot(Vmax=0.1, a=0, b=15, x=x)

    #HK = generateH(dx, m, hbar, Nx)

    #H = HK + np.diag(V)
    
    HKsp = generateHsp(dx, m, hbar, Nx)
    
    H = HKsp + sp.dia_matrix((V, [0]), shape=(Nx, Nx), dtype=np.complex64)
    
    # eVals, eVecs = SLA.eigsh(H, k =min(clip, Nx - 2), sigma=0, return_eigenvectors=True)
    
    t0 = time.time()

    for _ in range(Ni):
        #eVals = LA.eigh( H, eigvals_only=True, subset_by_index=[0, min(clip, Nx) - 1] )
        #eVals, eVecs = SLA.eigsh(H, k =(min(clip, Nx-1), sigma=0, return_eigenvectors=True)
        #eVals, eVecs = eigGPU.solve_sp(H, k=min(clip, Nx-1), mu0=eVals, x0=eVecs, dtype=np.float64)
        eVals, eVecs = fftSolver(H, N=N, k=min(clip, Nx-1), sigma=0)
        
    dt = time.time() - t0
    idt = dt/Ni
    
    return dt, idt, eVals

tdata = []
xdata = []

t0 = time.time()

for n in range(0, Nr):
    Nx = NxArray[n]  #int( 10 + n + np.exp(n/k) )
    
    # clip = clipArray[n]
    
    # N = N_Array[n]
    
    dt, idt, eVals = runExperiment(hbar, m, w, Xa, Xb, Ni, Nx, clip, N)
    
    tdata.append(idt)
    xdata.append(Nx)
    
    # print(f"Nx: {Nx}, dt: {dt}")
    print(f"Run Number: {n}\n Nx: {Nx}, Clip: {clip}, dt: {dt}")
    
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



plt.plot(xdata, tdata )

plt.xlabel("Number of Eigenvalues")
plt.ylabel("Time to solve / s")




























