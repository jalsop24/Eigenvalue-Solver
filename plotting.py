# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 21:47:28 2021

@author: James
"""



import numpy as np
import matplotlib.pyplot as plt



experimentName = "Harmonic 1D Nx = 10 - 4000 sparse eigs only"

inDir = "./data/" + experimentName

plotType = "sqrt"

xData = np.loadtxt(inDir + "/Nx_values.txt")
tData = np.loadtxt(inDir + "/dt_values.txt")



if plotType == "normal":
    plt.plot(xData, tData )
    plt.xlabel("Nx")
    plt.ylabel(" Time to solve / s ")
elif plotType == "sqrt":
    plt.plot(xData, np.sqrt(tData) )
    plt.xlabel("Nx")
    plt.ylabel(" $ (T / s)^{0.5} $ ")
elif plotType == "log":
    plt.plot(xData, np.log(tData) )
    plt.xlabel("Nx")
    plt.ylabel(" $ ln(T / s) $ ")










