# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:46:52 2021

@author: James
"""

import numpy as np
import scipy.sparse as sp
from scipy.fft import fft, fftn, ifft, ifftn, fftshift
import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D
import math
# from numba import jit
import scipy

from time import time

workers = 12

def FT(v, **kwargs):
    return fft(v, norm="ortho", workers=workers, **kwargs) 

def iFT(v, **kwargs):
    return ifft(v, norm="ortho", workers=workers, **kwargs) 


def FTn(v, **kwargs):
    return fftn(v, norm="ortho", workers=workers, **kwargs) 

def iFTn(v, **kwargs):
    return ifftn(v, norm="ortho", workers=workers, **kwargs) 


def segment(array, n):
    half = int(array.shape[0]/2)
    return np.array( array[ half - n : half + n ] )


def pad(array, finalSize):
    nZeros = finalSize - array.shape[0]
    
    shape = [finalSize]
    
    for i in range( 1, len(array.shape) ):
        shape.append(array.shape[i])
        
    zeros = np.zeros( shape , dtype=array.dtype)
    
    start = int( math.floor(nZeros/2) )
    
    i = 0
    for val in array:
        
        zeros[start + i] = val
        
        i += 1
        
    return zeros

def transformH_old(H, N):
    
    # assert( H.shape[0] == H.shape[1], "H must be a square matrix.")
    
    size = H.shape[0] 
    
    HFFT = np.zeros( shape=(N, N) , dtype=np.complex64)
    
    #iMat = sp.dia_matrix()
    
    for i in range(N):
        iVec = np.zeros(N, dtype=np.complex64)
            
        iVec[i] = 1

        iVec = pad(iVec, size)
        
        iVec = fftshift(iVec)
        
        iVec = np.transpose(iVec)
        
        for j in range(N):

            vec = np.zeros(N, dtype=np.complex64)
            vec[j] = 1
            
            vec = pad(vec, size)
            
            vec = fftshift(vec)
            
            # return  H @ iFT( vec ) 
            HFFT[i, j] = iVec @ FT( H @ iFT( vec ) )
    
    return HFFT

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


def transformH(H, N, p=1):
    
    # assert( H.shape[0] == H.shape[1], "H must be a square matrix.")
    
    t0 = time()
    
    size = H.shape[0] 
    
    hSize = int(N**p)
    
    HFFT = np.zeros( shape=(hSize, hSize) , dtype=np.complex64)
    
    if p == 1:
        
        iVec = np.ones( int(size) )
        
        data = [iVec, iVec]
        
        offsets = [-int(N/2), int(size - N/2)]
        
        iMat = sp.dia_matrix( (data, offsets), shape=(N, size) ).tocsr()
        
        jMat = iMat.transpose().toarray() # sp.dia_matrix( ([iVec, iVec], [-int(N/2), int(size - N/2)]), shape=(size, N) )
        
        t1 = time()
        
        ift = iFT( jMat , axis = 0)
        
        t2 = time()
        
        mult = H @ ift
        
        t3 = time()
        
        ftMult = FT( mult, axis = 0 )
        
        t4 = time()
        
        HFFT = iMat @ ftMult
        
        t5 = time()
        
        dt0 = t1 - t0
        dt1 = t2 - t1
        dt2 = t3 - t2
        dt3 = t4 - t3
        dt4 = t5 - t4
        
        times = [dt0, dt1, dt2, dt3, dt4]
    
        return HFFT, times
    
    elif p == 2:
        
        Nx = int(np.sqrt(size))
        
        t0 = time()
        
        fftMat1 = fftMat(Nx, N)
        
        t1 = time()
          
        mult = H @ fftMat1
        
        t2 = time()
        
        fftMat2 = fftMat1.transpose().conj()
        
        print(fftMat2.shape)
        
        t3 = time()
        
        HFFT = fftMat2 @ mult # Significant expense
        
        t4 = time()
        
        dt0 = t1 - t0
        dt1 = t2 - t1
        dt2 = t3 - t2
        dt3 = t4 - t3
        dt4 = 0
        
        times = [dt0, dt1, dt2, dt3, dt4]
        
        return HFFT, times
        

def transformHnew(H, N, p):
    
    HFFT = FT( FT( H.toarray(), n=N ).transpose().conj(), n=N )

    HFFT = fftshift(HFFT, axes=(0, 1))
    
    return HFFT
    
    

def solve(H, N, p, **kwargs):
    
    size = H.shape[0]
    
    HFFT, times = transformH(H, N, p)
    
    #print(HFFT)
    
    t0 = time()
    
    #eVals, eVecsK = sp.linalg.eigsh(HFFT, **kwargs)
    
    eVals, eVecsK = scipy.linalg.eigh(HFFT, **kwargs)
    
    t1 = time()
    
    eVecs = None
    
    if p == 1:
        
        eVecsK = fftshift(  pad( eVecsK , size) , axes=0)
    
        eVecs = iFT(eVecsK, axis=0)
    
    else:
        
        Nx = int(np.sqrt(size))
        
        eVecs = fftMat(Nx, N) @ eVecsK
    
    
    times.append(t1-t0)
    
    return eVals, eVecs, times
    
    
    
    





























