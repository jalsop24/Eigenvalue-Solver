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


outDir = "./data/"

experimentName = "test"

m = 1

Xa = -40
Xb = 40

Ni = 20

hbar = 1

w = 0.07

Nr = 50 # Number of runs to carry out

clip = 50 # First "clip" eigenvalues will be found

maxNx = 100_000

k = Nr/np.log(maxNx - 11 - Nr)

eVals = None 

def runExperiment(hbar, m, w, Xa, Xb, Ni, Nx, clip):
    
    x = np.linspace(Xa,Xb,Nx)
    dx = (Xb-Xa)/(Nx-1)
    
    V = 0.5 * m * w**2 * x**2

    #HK = generateH(dx, m, hbar, Nx)

    #H = HK + np.diag(V)
    
    HKsp = generateHsp(dx, m, hbar, Nx)
    
    Hsp = HKsp + sp.dia_matrix((V, [0]), shape=(Nx, Nx), dtype=np.complex64)
    
    #eVals, eVecs = LA.eigh( H, eigvals_only=False, subset_by_index=[0, min(clip, Nx) - 1] )
    
    t0 = time.time()

    for i in range(Ni):
        # eVals = LA.eigh( H, eigvals_only=True, subset_by_index=[0, min(clip, Nx) - 1] )
        eVals = SLA.eigsh(Hsp, k =(min(clip, Nx) - 2), sigma=0, return_eigenvectors=False)
    
    dt = time.time() - t0
    idt = dt/Ni
    
    return dt, idt, eVals

tdata = []
xdata = []

t0 = time.time()

for n in range(0, Nr):
    Nx = int( 10 + n + np.exp(n/k) )

    dt, idt, eVals = runExperiment(hbar, m, w, Xa, Xb, Ni, Nx, clip)
    
    tdata.append(idt)
    xdata.append(Nx)
    
    print(f"Nx: {Nx}, dt: {dt}")
    
dt = time.time() - t0



print(f"{Nr} experiments completed.")
print(f"Total time: {dt}")

makepath(experimentName)


xdata = np.array(xdata, dtype=np.int)
tdata = np.array(tdata)


with open(outDir + experimentName + "/params.txt", "w+") as file:
    file.write( f"Xa = {Xa},\nXb = {Xb},\nNi = {Ni},\ndt = {dt},\nw = {w},\nclip = {clip}" ) 

np.savetxt(outDir + experimentName + "/Nx_values.txt", xdata)
np.savetxt(outDir + experimentName + "/dt_values.txt", tdata)



plt.plot(xdata, tdata )

plt.xlabel("Nx")
plt.ylabel("Time to solve / s")




























