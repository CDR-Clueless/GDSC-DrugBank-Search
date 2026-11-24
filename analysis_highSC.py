#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 19 Nov 2026

@author: jds40
"""

from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm

from file_imports import import_allbyall

THRESH: float = 1.645
CRITICAL_PROP: float = 0.8

def main():
    consistencyHistogram()

def consistencyHistogram():
    dxg = import_allbyall()
    # Convert to Z-Scores
    for col in dxg.columns:
        dxg[col] = np.divide(np.array(dxg[col].values, dtype = float) - np.mean(dxg[col].values), np.std(dxg[col].values))
    # Go through linspace of numbers from 0 to 1 and get the bin sizes for each
    print(dxg.shape)
    xs = np.linspace(0., 1., num = 21, endpoint=True)
    ys = np.zeros(xs.shape[0]-1, dtype = int)
    # Get DataFarme of which SC values are above the threshold
    above = dxg >= THRESH
    # Get a list of proportions - each proportion is for a given gene, and is for how many drugs this gene is >= THRESH
    props = np.array([np.divide(np.sum(above.loc[ind]), len(above.loc[ind])) for ind in above.index], dtype = float)
    ys[-1] = np.sum(props >= xs[-1])
    for i in range(ys.shape[0]-1)[::-1]:
        # Get the proportion this entry must be above
        x = xs[i]
        ys[i] = np.sum(props >= x) - np.sum(ys[i+1:])
    print(xs)
    print(ys)
    print(np.sum(ys))
    plt.stairs(ys, xs)
    plt.xlabel("Proportion of Drug distributions in which gene has Z >= 1.645")
    plt.ylabel("Number of genes")
    plt.title("Consistency of Genes with High Z-Scores")
    plt.show()

def highestSC():
    dxg = import_allbyall()
    # Convert to Z-Scores
    for col in dxg.columns:
        dxg[col] = np.divide(np.array(dxg[col].values, dtype = float) - np.mean(dxg[col].values), np.std(dxg[col].values))
    above = dxg >= THRESH
    props = np.array([np.divide(np.sum(above.loc[ind]), len(above.loc[ind])) for ind in above.index], dtype = float)
    propsAbove = props >= CRITICAL_PROP
    genesAbove = dxg.index[propsAbove]
    genesAbove = "\n".join(genesAbove)
    with open("ConsistentHighGenes.txt", "w") as f:
        f.write(genesAbove)
    #plt.boxplot(props, whis = (0.01, 0.99))
    #plt.show()
    return

if(__name__=="__main__"):
    main()
