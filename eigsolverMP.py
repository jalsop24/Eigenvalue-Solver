# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 20:50:22 2021

@author: James
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
from src.makepath import makepath
import multiprocessing

outDir = "./data/"

experimentName = "Harmonic 1D Nx = 1000 MP"

Nx = 1000

m = 1

Xa = -40
Xb = 40
dx = (Xb-Xa)/(Nx-1)

Ni = 500

hbar = 1

processes = []

cores = 4

x = np.linspace(Xa,Xb,Nx)

#generate Hamiltonian matrix for eigenfunction solving

#Init Hamiltonian
H = np.zeros([Nx**2], dtype=np.complex64)

# Define total hamiltonian
H=np.reshape(H, [Nx, Nx])


# Kinetic part of the Hamiltonian
HK=np.reshape(H, [Nx, Nx])

hbar2 = hbar ** 2 #hbar squared
w = 0.07

hk = hbar2 / (2 * m * dx ** 2)

# print("Init H")

for k in range(0, Nx):# Go over each row
    #fill Hamiltonian, making use of hermiticity to only fill half and do conjugate
 
    HK[k][k] += 2 * hk # on-site kinetic part

    pmone= k -1
    if (pmone>=0 and pmone < Nx):
        HK[k][k-1] -= hk

    pmone= k+1
    if (pmone>=0 and pmone < Nx):
        HK[k][k+1] -= hk



def Vfunc(t):
    
    factor = (1 - 0.001*t)
    
    return np.outer(factor,  0.5 * w**2 * x**2)
  

V = 0.5 * m * w**2 * x**2


H = HK + np.diag(V)

t0 = time.time()
output = []

def solve(M):
    
    return LA.eigh(M)



if __name__ == "__main__":
    
    # for i in range(Ni):
    #     p = multiprocessing.Process(target=solve, args=(H,))
    #     processes.append(p)
    #     p.start()
    
    
    # for process in processes:
    #     process.join()
    
    hArray = []
    
    for i in range(Ni):
        hArray.append(H)
        
    pool = multiprocessing.Pool()
    result = pool.imap(solve, hArray, 20)
    
    pool.close()
    
    # t1 = time.time()
    # m = 0
    # for i in next(result):
    #     #print(i)
    #     output.append(i)
    #     m += 1
    #     if m > Ni:
    #         break
        
    # print(time.time() - t1)
    
    eVals, eVecs = LA.eigh( H )
    
    
    dt = time.time() - t0
    idt = dt/Ni
    
    print(f"{Ni} iterations completed.")
    print(f"Total time: {dt}")
    print(f"Average time per iteration: {idt}")
    
    makepath(experimentName)
    
    with open(outDir + experimentName + "/params.txt", "w+") as file:
        file.write( f"Nx = {Nx},\nXa = {Xa},\nXb = {Xb},\nNi = {Ni},\nidt = {idt},\ndt = {dt},\nw = {w}" ) 
        
    np.savetxt(outDir + experimentName + "/values.txt", eVals )
    
    n = 20
    
    fig = plt.figure(dpi=400, figsize=(16, 7))
    
    fig.suptitle(experimentName)
    
    nArray = np.arange(Nx)
    
    ax1 = fig.add_subplot(1, 2, 1, aspect=(Nx*hbar*w/np.max(eVals))  )
    ax1.plot( eVals/(hbar*w) )
    ax1.plot( (nArray + 0.5) )
    ax1.set_xlabel("n")
    ax1.set_ylabel("Energy/ $\hbar \omega$")
    ax1.set_title("Energy Eigenvalues")
    
    ax2 = fig.add_subplot(1, 2, 2, aspect=100)
    ax2.plot(x, eVecs[:,n].real)
    ax2.plot(x, V)
    ax2.set_ylim(-0.2, 0.2)
    ax2.set_xlim(Xa, Xb)
    ax2.set_title(f"N = {n} Eigenfunction")
    ax2.set_xlabel("x/au")
    ax2.set_ylabel("$\Psi$")
    
    fig.savefig(outDir + experimentName + "/plot.png")































