# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:53:42 2021

@author: James
"""


import numpy as np
import scipy.sparse as sp
import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D



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
    


def generateHsp(dx, m, hbar, Nx):
    
    hbar2 = hbar ** 2 
    hk = hbar2 / (2 * m * dx ** 2)
    
    base = np.ones(Nx, dtype=np.complex64)
    
    data = np.array([ -hk*base, 2*hk*base, -hk*base ])
    
    offsets = np.array([-1, 0, 1])
    
    H = sp.dia_matrix((data, offsets), shape=(Nx, Nx))
            
    return H
    
    

def genH2P(dx, m, hbar, Nx):
    
    hbar2 = hbar ** 2 
    hk = hbar2 / (2 * m * dx ** 2)
    
    dim = 1
    Np = 2
    
    sizes = (Nx, Nx)
    
    inds = np.zeros(dim*Np)
    
    PsiSize = (Nx) ** Np
    
    H = np.zeros([PsiSize**2], dtype=np.float32)
    
    for k in range(0, PsiSize):# Go over each row
        ind = k + k * PsiSize
        indPMone=0
        index1DtoND.index(k, inds, sizes, dim=dim, p=Np)

		# fill Hamiltonian, making use of hermiticity to only fill half and do conjugate

		#on-site potential
        # H[ind] += V.pot[k] # !
        
        for d in range(0, dim):
            #print("d", d)
            for p in range(0, Np):
                H[ind] += 2*hk # on-site kinetic part
                
                inds[p*dim + d] -= 1
                indPMone = indexNDto1D.index(inds, sizes, dim=dim, p=Np)
                
                if (inds[p*dim + d] >= 0 and inds[p*dim+d] < Nx):
                    H[int(k*PsiSize + indPMone)] -= hk
                    
                inds[p*dim + d] += 2
                indPMone = indexNDto1D.index(inds, sizes, dim=dim, p=Np)
                
                if (inds[p*dim+d]>=0 and inds[p*dim+d]< Nx and (k+1)<PsiSize):
                    H[int(k*PsiSize + indPMone)] -= hk
                    
                inds[p*dim+d]-=1

    H=np.reshape(H, [PsiSize,PsiSize])
    
    return H



def genH2Psp(dx, m, hbar, Nx, Vi, Vc):
    
    dataType = np.complex64
    
    hbar2 = hbar ** 2 
    hk = hbar2 / (2 * m * dx ** 2)

    Np = 2
    
    PsiSize = (Nx) ** Np
    
    V2 = np.zeros(PsiSize, dtype=dataType)
    
    cap = 1000
    
    for p in range(0, 2):
        for x1 in range(0, Nx):
            for x2 in range(0, Nx):
                    
                ind = indexNDto1D.index([x1, x2], (Nx, Nx), dim=1, p=2)
                    
                V2[ind] = Vi[x1] + 1.05*Vi[x2] + Vc(x2 - x1)
    
    
    base = hk*np.ones(PsiSize, dtype=dataType)
    
    data = np.array([ -base, -base, 4*base + V2, -base, -base ])
    
    offsets = np.array([-Nx, -1, 0, 1, Nx])
    
    H = sp.dia_matrix((data, offsets), shape=(PsiSize, PsiSize))
    
    return H, V2








































