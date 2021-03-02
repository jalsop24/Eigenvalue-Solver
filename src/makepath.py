# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 22:16:01 2021

@author: James
"""

import os

def makepath(name):
    if not os.path.isdir("./data"):
        os.mkdir("./data")

    dirPath = "./data/" + name
    if not os.path.isdir(dirPath):
        os.mkdir(dirPath)
        
    # dirPath = dirPath + "/waves"
    # if not os.path.isdir(dirPath):
    #     os.mkdir(dirPath)