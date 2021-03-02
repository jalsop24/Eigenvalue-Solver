

from scipy import linalg as LA
import numpy as np
from skcuda import linalg
from time import time
import pycuda.gpuarray as gpuarray


def single_solve(H, mu, b, error=0.001):
    
    
    H = np.matrix(H)
    b = np.matrix(b).transpose()
    
    mu_temp = mu
    
    for _ in range(10):
        
        step = LA.inv( (H - mu) ) * b
    
        magnitude = LA.norm(step)
        
        b = step/magnitude
        
        mu = b.getH() * H * b / LA.norm(b)
        
        terror = abs(mu - mu_temp)
        
        if terror < error:
            #print("within error")
            break
        else:
            mu_temp = mu
            #print(terror)
    
    return np.real(mu), b.transpose()
    
    

def solve(H, eValArray, eVecArray, error=0.001, clip=-1):
    
    # eValArray, eVecArray = LA.eigh( H )
    
    H_gpu = gpuarray.to_gpu(H)
    
    linalg.init()
    
    index = len(eValArray)
    
    if clip > 0:
        index = clip
    
    #print("ind", index)
    
    for i in range(index):
        mu, b = eValArray[i], eVecArray[i]
        
        #t0 = time()
        
        eValArray[i], eVecArray[i] = single_solve(H_gpu, mu, b, error=error)
        
       # print(time() - t0)
    
    return eValArray, eVecArray
    











