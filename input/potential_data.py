# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:39:49 2021

@author: James
"""


import numpy as np
import matplotlib.pyplot as plt

filename = "input/bandedge_Gamma_2d_rectangle_z_105"

if __name__ == "__main__":
    filename = "bandedge_Gamma_2d_rectangle_z_105"

offset = 596

data = np.loadtxt(filename+".dat")

coords = np.loadtxt(filename+".coord")

xArray = coords[:offset]

xMin = xArray[0]
xMax = xArray[-1]

yArray = coords[offset:]

yMin = yArray[0]
yMax = yArray[-1]

potential = np.reshape(data, (yArray.size, xArray.size)).transpose()

def pointEval(x, y):
    
    alphaX = (x - xMin)/(xMax - xMin) * len(xArray)
    alphaY = (y - yMin)/(yMax - yMin) * len(yArray)
    
    intX = np.int32( np.floor(alphaX) )
    fracX = alphaX - intX
    
    intY = np.int32( np.floor(alphaY) )
    fracY = alphaY - intY
    
    #print(intX, intY)
    
    p0 = potential[intX, intY]
    p1 = potential[intX+1, intY]
    p2 = potential[intX, intY+1]
    
    return (1-fracX-fracY)*p0 + fracX*p1 + fracY*p2


def get_raw_data():
    return potential



# plt.contourf(xArray, yArray, potential.transpose(), 25, cmap="inferno")










