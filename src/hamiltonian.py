# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:53:42 2021

@author: James
"""


import numpy as np
import scipy.sparse as sp
import pylops

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

from scipy.sparse.linalg import LinearOperator

from src.get_ccbin import get_ccbin

import math

import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D


ccbin = get_ccbin()



with open('./src/kernel.cu', 'r') as file:
    source = file.read()

# print("sys", os.system("cl"))

# Compile kernel using nvcc
mod = SourceModule(source, options=['-ccbin', ccbin])

gpu_product = mod.get_function("gpu_product")



# x_gpu, y_gpu, np.float64(factor), np.int32(Nx), V_gpu


def zeroPot(x):
    return 0

def derivativeMatrix(N, dtype=np.float64):
    base = np.ones(N, dtype=dtype)
    
    data = np.array([ -base, 2*base , -base ])
    
    offsets = np.array([-1, 0, 1])
    
    D = sp.dia_matrix((data, offsets), shape=(N, N))
    
    return D


def generateH(dx, m, hbar, Nx):
    '''
    

    Parameters
    ----------
    dx : float
        DESCRIPTION.
    m : float
        DESCRIPTION.
    hbar : float
        DESCRIPTION.
    Nx : int
        DESCRIPTION.

    Returns
    -------
    H : (Nx by Nx) nd-array.
        DESCRIPTION.

    '''
    
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
    


def generateHsp(dx, m, hbar, Nx, V, dtype=np.float32):
    
    '''
    

    Parameters
    ----------
    dx : float
        DESCRIPTION.
    m : float
        DESCRIPTION.
    hbar : float
        DESCRIPTION.
    Nx : int
        DESCRIPTION.

    Returns
    -------
    H : (Nx by Nx) nd-array.
        DESCRIPTION.

    '''
    
    hbar2 = hbar ** 2 
    hk = hbar2 / (2 * m * dx ** 2)
    
    D = derivativeMatrix(Nx, dtype=dtype)
    
    H = hk*D + sp.diags(V)
            
    return H


def generateHtd(dx, m, hbar, Nx, V, dtype=np.float32):
    
    '''
    

    Parameters
    ----------
    dx : float
        DESCRIPTION.
    m : float
        DESCRIPTION.
    hbar : float
        DESCRIPTION.
    Nx : int
        DESCRIPTION.

    Returns
    -------
    H : (Nx by Nx) nd-array.
        DESCRIPTION.

    '''
    
    hbar2 = hbar ** 2 
    hk = hbar2 / (2 * m * dx ** 2)
    
    D = np.ones(Nx, dtype=dtype) * -2.0*hk + V
    
    E = np.ones(Nx-1, dtype=dtype) * hk
          
    return D, E


# def genH2Psp(dx, m, hbar, Nx, Vi, Vc=zeroPot, dtype=np.complex64):
    
    

#     hbar2 = hbar ** 2 
#     hk = hbar2 / (2 * m * dx ** 2)

#     Np = 2
    
#     PsiSize = (Nx) ** Np
    
#     V2 = np.zeros(PsiSize, dtype=dtype)
    
    
#     for x1 in range(0, Nx):
#         for x2 in range(0, Nx):
                    
#             ind = indexNDto1D.index([x1, x2], (Nx, Nx), dim=1, p=2)
                    
#             V2[ind] = Vi[x1] + 1.00*Vi[x2] + Vc(x2 - x1) # Maybe bias one particle over the other to lift degeneracy
    
    
#     base = hk*np.ones(PsiSize, dtype=dtype)
    
#     data = np.array([ -base, -base, 4*base + V2, -base, -base ])
    
#     offsets = np.array([-Nx, -1, 0, 1, Nx])
    
#     H = sp.dia_matrix((data, offsets), shape=(PsiSize, PsiSize)).tocsr()
    
#     return H


def genH2Psp(dx, m, hbar, Nx, Vi, Vc=zeroPot, dtype=np.complex64):

    
    '''
    

    Parameters
    ----------
    dx : float
        DESCRIPTION.
    m : float
        DESCRIPTION.
    hbar : float
        DESCRIPTION.
    Nx : int
        DESCRIPTION.

    Returns
    -------
    H : (Nx by Nx) nd-array.
        DESCRIPTION.

    '''
    
    hbar2 = hbar ** 2 
    hk = hbar2 / (2 * m * dx ** 2)

    Np = 2
    
    PsiSize = (Nx) ** Np
    
    V2 = np.zeros(PsiSize, dtype=dtype)
    
    if Vc is None:
        Vc = zeroPot
    
    for x1 in range(0, Nx):
        for x2 in range(0, Nx):
                    
            ind = indexNDto1D.index([x1, x2], (Nx, Nx), dim=1, p=2)
                    
            V2[ind] = Vi[x1] + 1.00*Vi[x2] + Vc(x2 - x1) # Maybe bias one particle over the other to lift degeneracy
    
    V2 = sp.diags(V2)
    
    Dx = derivativeMatrix(Nx, dtype=dtype)
    
    L = sp.kronsum(Dx, Dx)
    
    H = hk*L + V2
    
    return H

def genH2Dsp(dx, dy, Nx, Ny, m, hbar, Vi):

    
    '''
    

    Parameters
    ----------
    dx : float
        DESCRIPTION.
    m : float
        DESCRIPTION.
    hbar : float
        DESCRIPTION.
    Nx : int
        DESCRIPTION.

    Returns
    -------
    H : (Nx by Nx) nd-array.
        DESCRIPTION.

    '''
    
    dataType = np.complex64
    
    hbar2 = hbar ** 2 
    hkx = hbar2 / (2 * m * dx **2 )
    hky = hbar2 / (2 * m * dy **2 )
    
    PsiSize = (Nx * Ny)
    
    # V2 = np.zeros(PsiSize, dtype=dataType)
    
    base = np.ones(PsiSize, dtype=dataType)
    
    data = np.array([ -hky*base, -hkx*base, 2*(hkx + hky)*base + Vi, -hkx*base, -hky*base ])
    
    offsets = np.array([-Nx, -1, 0, 1, Nx])
    
    H = sp.dia_matrix((data, offsets), shape=(PsiSize, PsiSize)).tocsr()
    
    return H
    
    

def generateHLOP(dx, m, hbar, Nx, V, **kwargs):

    
    '''
    

    Parameters
    ----------
    dx : float
        DESCRIPTION.
    m : float
        DESCRIPTION.
    hbar : float
        DESCRIPTION.
    Nx : int
        DESCRIPTION.

    Returns
    -------
    H : (Nx by Nx) nd-array.
        DESCRIPTION.

    '''
    
    kinetic = pylops.SecondDerivative(Nx, sampling=1, **kwargs  ) # (2*m*dx/(hbar**2))
    
    H = kinetic #+ pylops.Diagonal(V, **kwargs)
    
    # H = LinearOperator(shape=(Nx, Nx), matvec=matvec)
    
    return H





def getLOP(dx, m, hbar, Nx, V):

    
    '''
    

    Parameters
    ----------
    dx : float
        DESCRIPTION.
    m : float
        DESCRIPTION.
    hbar : float
        DESCRIPTION.
    Nx : int
        DESCRIPTION.

    Returns
    -------
    H : (Nx by Nx) nd-array.
        DESCRIPTION.

    '''
    
    blockSize = 256
    
    factor = np.float64(hbar**2/2*m*dx**2)
    Nx = np.int32(Nx)
    
    numBlocks = int( np.ceil( Nx / blockSize ) )
    
    factorGPU = cuda.mem_alloc_like(factor)
    cuda.memcpy_htod(factorGPU, factor)
    
    NxGPU = cuda.mem_alloc_like(Nx)
    cuda.memcpy_htod(NxGPU, Nx)
    
    V_gpu = gpuarray.to_gpu(V)
    
    # gpu_product.prepare( [np.intp, np.intp, np.float64, np.int32, np.intp] )
    
    def matvec(x):
        
        y = np.zeros_like(x)
        
        # x_gpu = cuda.mem_alloc_like(x)
        # cuda.memcpy_htod(x_gpu, x)
        
        # y_gpu = cuda.mem_alloc_like(y)
        # cuda.memcpy_htod(x_gpu, y)
        
        x_gpu = gpuarray.to_gpu(x) 
        
        y_gpu = gpuarray.to_gpu(y)
        
        
        
        gpu_product( x_gpu, y_gpu, factor, Nx, V_gpu, block=(blockSize,1,1), grid=(numBlocks,1))  
        
        # gpu_product.prepared_call( (numBlocks,1), x_gpu.ptr, y_gpu.ptr, np.float64(factor), np.int32(Nx), V_gpu.ptr)
        
        y = y_gpu.get()
        
        return y
        
    
    newLOP = LinearOperator(shape=(Nx, Nx), matvec=matvec)

    return newLOP





























