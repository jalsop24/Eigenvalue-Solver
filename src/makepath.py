# -*- coding: utf-8 -*-
"""

Taken from pySL:

H. V. Lepage
A. A. Lasek
C. H. W. Barnes


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