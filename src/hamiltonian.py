# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:53:42 2021

@author: James
"""


import numpy as np
import scipy.sparse as sp
import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D

def zeroPot(x):
    return 0

def generateH(dx, m, hbar, Nx):
    
    hbar2 = hbar ** 2 
    hk = hbar2 / (2 * m * dx ** 2)
    
    H = np.zeros([Nx, Nx], dtype=np.complex64)
    
    for k in range(0, Nx):# Go over each row
        #fill Hamiltonian, making use of hermiticity to only fill half and do conjugate
     
        H[k][k] += 2 * hk # on-site kinetic part
    
        pmone= k -1
        if (pmone>=0 and pmone < Nx):
            H[k][k-1] -= hk
    
        pmone= k+1
        if (pmone>=0 and pmone < Nx):
            H[k][k+1] -= hk
            
    return H
    


def generateHsp(dx, m, hbar, Nx, dtype=np.float32):
    
    hbar2 = hbar ** 2 
    hk = hbar2 / (2 * m * dx ** 2)
    
    base = np.ones(Nx, dtype=dtype)
    
    data = np.array([ -hk*base, 2*hk*base, -hk*base ])
    
    offsets = np.array([-1, 0, 1])
    
    H = sp.dia_matrix((data, offsets), shape=(Nx, Nx))
            
    return H


def genH2Psp(dx, m, hbar, Nx, Vi, Vc=zeroPot):
    
    dataType = np.complex64
    
    hbar2 = hbar ** 2 
    hk = hbar2 / (2 * m * dx ** 2)

    Np = 2
    
    PsiSize = (Nx) ** Np
    
    V2 = np.zeros(PsiSize, dtype=dataType)
    
    
    for x1 in range(0, Nx):
        for x2 in range(0, Nx):
                    
            ind = indexNDto1D.index([x1, x2], (Nx, Nx), dim=1, p=2)
                    
            V2[ind] = Vi[x1] + 1.01*Vi[x2] + Vc(x2 - x1)
    
    
    base = hk*np.ones(PsiSize, dtype=dataType)
    
    data = np.array([ -base, -base, 4*base + V2, -base, -base ])
    
    offsets = np.array([-Nx, -1, 0, 1, Nx])
    
    H = sp.dia_matrix((data, offsets), shape=(PsiSize, PsiSize))
    
    return H, V2








































