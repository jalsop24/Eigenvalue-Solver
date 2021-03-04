################################################################################################################################
# Real space sparse eigensolver using the GPU
################################################################################################################################

import numpy as np
import ctypes
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import os
import scipy.sparse as sp
from skcuda import linalg



def solve_sp(H, dtype=np.float32, tol=1e-3): #Real space solver GPU

    
    PsiSize = H.shape[0]
    
    dataType = dtype
    
    #Do eig things here
   
    if os.name == 'nt':
       _libcusparse = ctypes.cdll.LoadLibrary('cusparse64_11.dll')
       _libcusolver = ctypes.cdll.LoadLibrary('cusolver64_11.dll')
    else:
        _libcusparse = ctypes.cdll.LoadLibrary('libcusparse.so')
        _libcusolver = ctypes.cdll.LoadLibrary('libcusolver.so')
        
    _libcusparse.cusparseCreate.restype = int
    _libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]

    _libcusparse.cusparseDestroy.restype = int
    _libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]

    _libcusparse.cusparseCreateMatDescr.restype = int
    _libcusparse.cusparseCreateMatDescr.argtypes = [ctypes.c_void_p]


    # cuSOLVER
    

    _libcusolver.cusolverSpCreate.restype = int
    _libcusolver.cusolverSpCreate.argtypes = [ctypes.c_void_p]

    _libcusolver.cusolverSpDestroy.restype = int
    _libcusolver.cusolverSpDestroy.argtypes = [ctypes.c_void_p]

    _libcusolver.cusolverSpDcsreigvsi.restype = int
    _libcusolver.cusolverSpDcsreigvsi.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_float,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_float,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p]

    
    # print(dataType)
    
    Hcsr = sp.csr_matrix(H, dtype=dataType)
    
    # print(Hcsr)
    
    x0 = gpuarray.to_gpu(np.ones(PsiSize, dtype=dataType))
    
    x = gpuarray.to_gpu(np.zeros(PsiSize, dtype=dataType))
    mu = gpuarray.to_gpu(np.zeros(1, dtype=dataType))
    
    Hmu = np.zeros(1, dtype=dataType)
    Hx = np.zeros(PsiSize, dtype=dataType)

    # Copy arrays to GPU
    dcsrVal = gpuarray.to_gpu(Hcsr.data)
    dcsrColInd = gpuarray.to_gpu(Hcsr.indices)
    dcsrIndPtr = gpuarray.to_gpu(Hcsr.indptr)


    # Create solver parameters
    m = ctypes.c_int(Hcsr.shape[0])  # Need check if A is square
    nnz = ctypes.c_int(Hcsr.nnz)
    descrA = ctypes.c_void_p()
    mu0 = ctypes.c_float(5)
    maxite = ctypes.c_int(100)
    tolerance = ctypes.c_float(tol)
   
    # create cusparse handle
    _cusp_handle = ctypes.c_void_p()
    status = _libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
    assert(status == 0)
    cusp_handle = _cusp_handle.value

    # create MatDescriptor
    status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(descrA))
    assert(status == 0)

    #create cusolver handle
    _cuso_handle = ctypes.c_void_p()
    status = _libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
    assert(status == 0)
    cuso_handle = _cuso_handle.value
    
    # print(_cuso_handle)
    
    # Solve
    status=_libcusolver.cusolverSpDcsreigvsi(cuso_handle,
                                         m,
                                         nnz,
                                         descrA,
                                         int(dcsrVal.gpudata),
                                         int(dcsrIndPtr.gpudata),
                                         int(dcsrColInd.gpudata),
                                         mu0,
                                         int(x0.gpudata),
                                         maxite,
                                         tolerance,
                                         int(mu.gpudata),
                                         int(x.gpudata))

    
    mu.get(Hmu)
    x.get(Hx)

    # print(status)
    

    # Destroy handles
    status = _libcusolver.cusolverSpDestroy(cuso_handle)
    assert(status == 0)
    status = _libcusparse.cusparseDestroy(cusp_handle)
    assert(status == 0)

   
    print(Hmu, Hx)
    
    return Hmu, Hx
	



def solve(H):
    
    
    # eValArray, eVecArray = LA.eigh( H )
    
    PsiSize = H.shape[0]
    
    H=np.reshape(H.ravel(), [PsiSize, PsiSize], order='F')
    
    D=np.zeros(PsiSize, dtype=np.float64)
    
    H_gpu = gpuarray.to_gpu(H)
    
    linalg.init()
    V_gpu, D_gpu = linalg.eig(H_gpu, 'N', 'V')
    
    print(D_gpu.dtype)
    
    V_gpu.get(H) 
    D_gpu.get(D)
    
    H=np.reshape(H.ravel(), [PsiSize, PsiSize], order='C')
    
    return D, H





























