# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 20:08:41 2021

@author: James
"""


import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

from scipy.sparse.linalg import LinearOperator


from src.get_ccbin import get_ccbin



ccbin = get_ccbin()



with open('./src/kernel.cu', 'r') as file:
    source = file.read()

# print("sys", os.system("cl"))

# Compile kernel using nvcc
mod = SourceModule(source, options=['-ccbin', ccbin, "-rdc",  "true"])

dot_product = mod.get_function("dot_product")

def getLOP(V, Nx, dx, m):
    
    V_gpu = gpuarray.to_gpu(V)
    
    def matmat(x):
        return 0
    
    newLOP = LinearOperator(shape=(Nx, Nx), matmat=matmat, matvec=matmat, )
    











