# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:45:17 2021

@author: James
"""

import numpy as np
# import scipy.sparse as sp
# import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D

class wavefunction:
    
    def  __init__(self, data, Nx, particles=1):
        
        self.Nx = Nx
        self.Particles = particles
        self.Data = data
       
        
    def __index__(self, x1, x2):
        
        ind = indexNDto1D.index((x1, x2), (self.Nx, self.Nx), dim=1, p=self.Particles)
        
        return self.Data[ind]
        
    def norm(self, dx):
        v = self.Data
        
        return np.sqrt( np.inner(np.conjugate(v), v) * dx )

    def normalise(self, dx):
        v = self.Data
        return wavefunction( v / self.norm(dx), self.Nx, self.Particles)
    
    def prob(self, n=None):
        v = self.Data
        if self.Particles == 1 or n == None:
            return v.conjugate() * v
        elif n == 1:
            return self.prob1()
        elif n == 2: 
            return self.prob2()
    
    def prob1(self):
        
        v = self.Data
        
        size =  int( np.sqrt(v.size) )
        
        p = np.zeros( size )
        
        #print(size)
        
        mod2 = self.prob()
        
        #print(mod2.size)
        
        mod2 = np.reshape(mod2, [size, size] )
        
        #print(mod2.shape)
        
        for i in range(size):
            p[i] = np.sum( mod2[i,:] )
        
        return p
    
    def prob2(self):
        
        v = self.Data
        
        size =  int( np.sqrt(v.size) )
        
        p = np.zeros( size )
        
        #print(size)
        
        mod2 = self.prob()
        
        #print(mod2.size)
        
        mod2 = np.reshape(mod2, [size, size] )
        
        #print(mod2.shape)
        
        for i in range(size):
            p[i] = np.sum( mod2[:,i] )
        
        return p
        