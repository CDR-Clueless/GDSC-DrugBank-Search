#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 22 Jul 2026

@author: jds40
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def main():
    fileLoc = os.path.join("Data", "Results", "GLM Model Results-High Variance")
    r2s, r2adjs = [], []
    for filename in os.listdir(fileLoc):
        if("stats" not in filename or "coefficients" in filename):
            continue
        with open(os.path.join(fileLoc, filename), "r") as f:
            mainList = f.read().split("\n")
        r2 = mainList[0].split("\t")[-1]
        r2adj = mainList[1].split("\t")[-1]
        if(r2!=""):
            r2s.append(r2)
        if(r2adj!=""):
            r2adjs.append(r2adj)
    # Make these lists into arrays
    r2 = np.array(r2s, dtype = float)
    r2adj = np.array(r2adjs, dtype = float)
    ## Make scatter plots
    #plt.scatter(["R2" for _ in range(r2.shape[0])], r2)
    #plt.scatter(["R2 Adj." for _ in range(r2adj.shape[0])], r2adj)
    #plt.show()
    ## Make Boxplots
    # Plot boxplot
    plt.boxplot([r2, r2adj], tick_labels = ["R2", "R2 Adj."])
    plt.ylabel("R Squared Value")
    plt.title("GLM Accuracy predicting drug response using High variance genes")
    plt.show()
    return

if(__name__=="__main__"):
    main()
