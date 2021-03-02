# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:45:17 2021

@author: James
"""

# import numpy as np
# import scipy.sparse as sp
# import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D

class wavefunction:
    
    def  __init__(self, Nx, particles, data):
        
        self.Nx = Nx
        self.Particles = particles
        self.Data = data
       
        
    def __index__(self, x1, x2):
        
        ind = indexNDto1D.index((x1, x2), (self.Nx, self.Nx), dim=1, p=self.Particles)
        
        return self.Data[ind]
        

    