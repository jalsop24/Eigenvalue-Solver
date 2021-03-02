

from skcuda import linalg
from time import time
import pycuda.gpuarray as gpuarray


    

def solve(H):
    
    # eValArray, eVecArray = LA.eigh( H )
    
    H_gpu = gpuarray.to_gpu(H)
    
    linalg.init()
    
    V_gpu, D_gpu = linalg.eig(H_gpu, 'N', 'V')
    
    return V_gpu.get(), D_gpu.get()











