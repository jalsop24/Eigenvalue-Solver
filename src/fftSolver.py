# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:46:52 2021

@author: James
"""

import numpy as np
import scipy.sparse as sp
from scipy.fft import fft, ifft, fftshift
import math
# from numba import jit
import scipy

scipy.fft.set_workers(4)

def FT(v, **kwargs):
    return fft(v, norm="ortho", **kwargs) 

def iFT(v, **kwargs):
    return ifft(v, norm="ortho", **kwargs) 


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


def transformH(H, N):
    
    # assert( H.shape[0] == H.shape[1], "H must be a square matrix.")
    
    size = H.shape[0] 
    
    HFFT = np.zeros( shape=(N, N) , dtype=np.complex64)
    
    iVec = np.ones( int(size) )
    
    iMat = sp.dia_matrix( ([iVec, iVec], [-int(N/2), int(size - N/2)]), shape=(N, size) )
    
    jMat = iMat.transpose().toarray() # sp.dia_matrix( ([iVec, iVec], [-int(N/2), int(size - N/2)]), shape=(size, N) )
    
    HFFT = iMat @ FT( H @ iFT( jMat , axis = 0), axis = 0 )

    return HFFT



def solve(H, N, **kwargs):
    
    size = H.shape[0]
    
    HFFT = transformH(H, N)
    
    eVals, eVecsK = sp.linalg.eigsh(HFFT, **kwargs)
    
    eVecsK = pad(eVecsK, size)
    
    eVecs = iFT(eVecsK, axis=0)
    
    return eVals, eVecs
    
    
    
    





























