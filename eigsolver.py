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
from src.makepath import makepath
from src.hamiltonian import generateH, generateHsp, genH2P, genH2Psp
import src.wavefunction as wf
import src.potential as potential
# from src.inversePower import solve
# from src.eigGPU import solve as solveGPU

outDir = "./data/"

experimentName = "test"

Nx = 1000

m = 1

Xa = -40
Xb = 40
dx = (Xb-Xa)/(Nx-1)

Ni = 1

hbar = 1

clip = 15

x = np.linspace(Xa,Xb,Nx)

w = 0.07

eVals, eVecs = None, None

# print("Init H")
 
    
def norm(v, dx):
    return np.sqrt( np.inner(np.conjugate(v), v) * dx )

def normalise(v, dx):
    return v / norm(v, dx)

def prob(v):
    return v.conjugate() * v

def prob1(v):
    
    size =  int( np.sqrt(v.size) )
    
    p = np.zeros( size )
    
    #print(size)
    
    mod2 = prob(v)
    
    #print(mod2.size)
    
    mod2 = np.reshape(mod2, [size, size] )
    
    #print(mod2.shape)
    
    for i in range(size):
        p[i] = np.sum( mod2[i,:] )
    
    return p

def prob2(v):
    
    size =  int( np.sqrt(v.size) )
    
    p = np.zeros( size )
    
    #print(size)
    
    mod2 = prob(v)
    
    #print(mod2.size)
    
    mod2 = np.reshape(mod2, [size, size] )
    
    #print(mod2.shape)
    
    for i in range(size):
        p[i] = np.sum( mod2[:,i] )
    
    return p
    
    
def coloumbPot(index_diff):
    
    distance = abs(index_diff * dx)
    
    e0 = 1
    
    if distance < dx:
        return 1/(4*np.pi*e0*dx)
    else:
        return 1/(4*np.pi*e0*(distance))
    

# h2p = genH2P(dx, m, hbar, Nx)


t0 = time.time()

V = 0.5 * m * w**2 * x**2

V = potential.doubleDot(Vmax=0.1, a=0, b=15, x=x)


# HK = generateH(dx, m, hbar, Nx)

# H = HK + np.diag(V)


t1 = time.time()

# H = generateHsp(dx, m, hbar, Nx) + sp.dia_matrix((V, [0]), shape=(Nx, Nx), dtype=np.complex64)

H, V2 = genH2Psp(dx, m, hbar, Nx, Vi = V, Vc=coloumbPot)


# eVals, eVecs = LA.eigh( H, eigvals_only=False, subset_by_index=[0, min(clip, Nx) - 1] )

# Hcsr = sp.csr_matrix(H)


t2 = time.time()

for i in range(Ni):
    # print(i)
    # eVals, eVecs = solve(H, eVals, eVecs, clip=50)
    # eVals, eVecs = solveGPU(H)
    # eVals, eVecs = LA.eigh( H, eigvals_only=False, subset_by_index=[0, min(clip, Nx) - 1] )
    eVals, eVecs = SLA.eigsh( H, k=min(clip, Nx-2), sigma=0, return_eigenvectors=True, tol = 0.01, ncv=int(2*clip))
    # eVals = SLA.eigsh( H, k=min(clip, Nx-2), sigma=0, return_eigenvectors=False)
    

t3 = time.time()

dt = t3 - t0
idt = (t3-t2)/Ni

dt1 = t1 - t0
dt2 = t2 - t1

print(f"{Ni} iterations completed.")
print(f"Total time: {dt}")
print(f"Average time per iteration: {idt}")
print(f"dt1 = {dt1},\ndt2 = {dt2}")

makepath(experimentName)

with open(outDir + experimentName + "/params.txt", "w+") as file:
    file.write( f"Nx = {Nx},\nXa = {Xa},\nXb = {Xb},\nNi = {Ni},\nidt = {idt},\ndt = {dt},\nw = {w}" ) 
    
np.savetxt(outDir + experimentName + "/values.txt", eVals )

n = 2

n = min(n, len(eVecs[1])-1)

fig = plt.figure(dpi=400, figsize=(16, 7))

fig.suptitle(experimentName)

nArray = np.arange(len(eVals))

ax1 = fig.add_subplot( 1, 2, 1 ) #  , aspect=( len(eVals)*hbar*w/np.max(eVals).real ) 
ax1.plot( eVals.real/(hbar*w) )
# ax1.plot( (nArray + 0.5) )
ax1.set_xlabel("n")
ax1.set_ylabel("Energy/ $\hbar \omega$")
ax1.set_title("Energy Eigenvalues")

# a = m*w/hbar
# y = np.sqrt(a) * x
# eig3 = np.power(a/np.pi, 0.25) * np.power(3, -0.5) * ( 2*y**3 - 3*y ) * np.exp(-0.5*y**2 )

eigOut = eVecs[:,n]

# print( norm(eig3, dx) )
# print( norm(eigOut, dx))

eigOut = normalise(eigOut, dx)

# print( norm(eigOut, dx) )

ax2 = fig.add_subplot(1, 2, 2, aspect=100)
ax2.plot(x, prob1(eigOut) )
ax2.plot(x, prob2(eigOut) )
# ax2.plot(x, prob(eigOut) )
ax2.plot(x, V)
# ax2.plot(x, prob(eig3) )
ylim = 0.5
ax2.set_ylim(-0.05, ylim)
ax2.set_xlim(Xa, Xb)
ax2.set_title(f"N = {n} Eigenfunction, Nx = {Nx}")
ax2.set_xlabel("x/au")
ax2.set_ylabel("$ |\Psi|^2 $")

fig.savefig(outDir + experimentName + "/plot.png")































