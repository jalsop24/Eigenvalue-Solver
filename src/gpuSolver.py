"""

Taken from pySL 
H. V. Lepage
A. A. Lasek
C. H. W. Barnes

"""


################################################################################################################################
# Real space sparse eigensolver using the GPU
################################################################################################################################

import numpy as np
import ctypes
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import os
from time import time
import scipy.sparse as sp
from skcuda import linalg



def solve_sp(H, k=None, mu0=None, x0=None, dtype=np.float64, tol=1e-7): #Real space solver GPU
    
    eVals, eVecs = None, None

    PsiSize = H.shape[0]
    
    dataType = dtype
    
    # Do eig things here
   
    if os.name == 'nt':
        # print("a", os.name)
        _libcusparse = ctypes.cdll.LoadLibrary('cusparse64_10')
        _libcusolver = ctypes.cdll.LoadLibrary('cusolver64_10')
    else:
        # print("b")
        _libcusparse = ctypes.cdll.LoadLibrary('libcusparse.so')
        _libcusolver = ctypes.cdll.LoadLibrary('libcusolver.so')
        
    _libcusparse.cusparseCreate.restype = int
    _libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]

    _libcusparse.cusparseDestroy.restype = int
    _libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]

    _libcusparse.cusparseCreateMatDescr.restype = int
    _libcusparse.cusparseCreateMatDescr.argtypes = [ctypes.c_void_p]

    
    _libcusparse.cusparseSetMatFillMode.restype = int 
    _libcusparse.cusparseSetMatFillMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    
    _libcusparse.cusparseSetMatDiagType.restype = int 
    _libcusparse.cusparseSetMatDiagType.argtypes = [ctypes.c_void_p, ctypes.c_int]
    
    
    _libcusparse.cusparseGetMatFillMode.restype = int
    _libcusparse.cusparseGetMatFillMode.argtypes = [ctypes.c_void_p]
    
    _libcusparse.cusparseGetMatDiagType.restype = int
    _libcusparse.cusparseGetMatDiagType.argtypes = [ctypes.c_void_p]



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
    
    x = gpuarray.to_gpu(np.empty(PsiSize, dtype=dataType))
    mu = gpuarray.to_gpu(np.empty(1, dtype=dataType))
    
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
    maxite = ctypes.c_int(1000)
    tolerance = ctypes.c_float(tol)
    
    #create cusolver handle
    _cuso_handle = ctypes.c_void_p()
    status = _libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
    assert(status == 0)
    cuso_handle = _cuso_handle.value
    
    # create cusparse handle
    _cusp_handle = ctypes.c_void_p()
    status = _libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
    assert(status == 0)
    cusp_handle = _cusp_handle.value
    
    # create MatDescriptor
    status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(descrA))
    assert(status == 0)
    
    
    # _libcusparse.cusparseSetMatFillMode(descrA, ctypes.c_int(0) )
    # _libcusparse.cusparseSetMatDiagType(descrA, ctypes.c_int(0) )
    
    fMode = _libcusparse.cusparseGetMatFillMode(descrA)
    dMode = _libcusparse.cusparseGetMatDiagType(descrA)
    
    # print("f/d mode: ", fMode, dMode)
    
    # Solve
    
    t0 = time()
    
    if k is None:
        k = 1
    else:
        assert(len(mu0) >= k)
        assert(len(x0[0]) >= k)
    
    eVals = np.zeros(k, dtype=dataType)
    eVecs = np.zeros( (PsiSize, k), dtype=dataType )
    
    for i in range(0, k):
        
        # print(i)
        
        mu0_temp, x0_temp = None, None
        
        if mu0 is None:
            mu0_temp = ctypes.c_float(0)
        else:
            mu0_temp = ctypes.c_float(mu0[i])
        
        if x0 is None:
            _x0 = np.random.rand(PsiSize)
            x0_temp = gpuarray.to_gpu(_x0)
        else:
            x0_temp = gpuarray.to_gpu(x0[i])
        
        
        
        status=_libcusolver.cusolverSpDcsreigvsi(cuso_handle,
                                         m,
                                         nnz,
                                         descrA,
                                         int(dcsrVal.gpudata),
                                         int(dcsrIndPtr.gpudata),
                                         int(dcsrColInd.gpudata),
                                         mu0_temp,
                                         int(x0_temp.gpudata),
                                         maxite,
                                         tolerance,
                                         int(mu.gpudata),
                                         int(x.gpudata))
        
        mu.get(Hmu)
        x.get(Hx)
        
        np.append(eVals, Hmu)
        np.append(eVecs, Hx)
        
        
        
        
    # print(Hmu, Hx)
    
    #print(time()-t0)
    #print(mu.view())
    
    

    # print(status)
    

    # Destroy handles
    status = _libcusolver.cusolverSpDestroy(cuso_handle)
    assert(status == 0)
    status = _libcusparse.cusparseDestroy(cusp_handle)
    assert(status == 0)

   
    # print(Hmu, Hx)
    
    return eVals, eVecs
	



def solve(H):
    
    
    # eValArray, eVecArray = LA.eigh( H )
    
    PsiSize = H.shape[0]
    
    H=np.reshape(H.ravel(), [PsiSize, PsiSize], order='F')
    
    D=np.zeros(PsiSize, dtype=np.float32)
    
    H_gpu = gpuarray.to_gpu(H)
    
    linalg.init()
    V_gpu, D_gpu = linalg.eig(H_gpu, 'N', 'V')
    
    print(D_gpu.dtype)
    
    V_gpu.get(H) 
    D_gpu.get(D)
    
    H=np.reshape(H.ravel(), [PsiSize, PsiSize], order='C')
    
    return D, H





























