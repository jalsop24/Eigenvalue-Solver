# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 21:47:28 2021

@author: James
"""



import numpy as np
import matplotlib.pyplot as plt



experimentName = "FFT/Harmonic 1 Particle Nx = 500 - 89000"

inDir = "./data/" + experimentName

plotType = "normal"

xLabel = "Nx"



xData = np.loadtxt(inDir + "/x_values.txt")
tData = np.loadtxt(inDir + "/dt_values.txt")

if plotType == "normal":
    plt.plot(xData, tData )
    plt.xlabel(xLabel)
    plt.ylabel(" Time to solve / s ")
elif plotType == "sqrt":
    plt.plot(xData, np.sqrt(tData) )
    plt.xlabel(xLabel)
    plt.ylabel(" $ (T / s)^{0.5} $ ")
elif plotType == "log":
    plt.plot(xData, np.log(tData) )
    plt.xlabel(xLabel)
    plt.ylabel(" $ ln(T / s) $ ")


plt.savefig(inDir + "/plot-" + plotType + ".png")







