# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 18:06:17 2021

@author: James
"""

import numpy as np
import scipy.sparse as sp
import src.hamiltonian as hamiltonian
import src.potential as potential
from scipy.fft import fft, ifft, fftshift, fftn, ifftn
from time import time
from src.fftSolver import transformH, transformHnew
import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D

workers = 12

def FT(v, **kwargs):
    return fft(v, norm="ortho", workers=workers, **kwargs) 

def iFT(v, **kwargs):
    return ifft(v, norm="ortho", workers=workers, **kwargs) 


def FTn(v, **kwargs):
    return fftn(v, norm="ortho", workers=workers, **kwargs) 

def iFTn(v, **kwargs):
    return ifftn(v, norm="ortho", workers=workers, **kwargs) 


Nx = 20

m = 1

Xa = -50
Xb = 50
dx = (Xb-Xa)/(Nx-1)

hbar = 1

clip = 50

N = 10

w = 0.07

# x = np.linspace(Xa, Xb, Nx, dtype=np.float64)

size = int(Nx**2)
nSize = int(N**2)

def fftMat(Nx, N):
    
    size = int(Nx**2)
    
    nSize = int(N**2)
    
    fftMat = np.zeros( (size, nSize), dtype=np.complex64 )

    k1Mat = np.zeros((Nx, Nx))

    for i in range( nSize ):
    
        k1Mat = np.zeros(nSize)
        
        k1Mat[i] = 1
        
        k1Mat = k1Mat.reshape(N, N)
        
        paddedMat = np.zeros( shape=(Nx, Nx) )
        
        index = int((Nx-N)/2)
        
        paddedMat[index:N+index, index:N+index] = k1Mat
        
        paddedMat2 = fftshift(paddedMat)
        
        fftMat[:, i] = iFTn( paddedMat2 ).reshape(size)

    
    return fftMat


    
fftMat1 = fftMat(Nx, N)


x = np.ones(N)

testVec = np.zeros((N, N))

testVec = fftshift( np.diag(x) )

testVec0 = testVec.reshape(nSize)

outVec = fftMat1 @ testVec0
outVec = outVec.reshape((Nx, Nx))

outVec2 = iFTn(testVec)



i = 15

nSize = int(N**2)

k1Mat = np.zeros(nSize)

for i in range(nSize):

    k1Mat[i] = i+1
        
k1Mat = k1Mat.reshape(N, N)
        
paddedMat = np.zeros(shape=(Nx, Nx) )
        
index = int((Nx-N)/2)
        
paddedMat[index:N+index, index:N+index] = k1Mat

paddedMat2 = fftshift(paddedMat)

ftMat = iFTn( paddedMat2 )

