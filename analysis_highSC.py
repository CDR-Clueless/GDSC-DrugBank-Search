#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 19 Nov 2026

@author: jds40
"""

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
    xs = np.linspace(0., 1., num = 21, endpoint=True)
    ys = np.zeros(xs.shape[0]-1, dtype = int)
    for i in tqdm(range(ys.shape[0])[::-1]):
        x = xs[i]
        above = dxg >= THRESH
        props = np.array([np.divide(np.sum(above.loc[ind]), len(above.loc[ind])) for ind in above.index], dtype = float)
        propsAbove = props >= x
        if(i + 1 < ys.shape[0]):
            ys[i] = len(dxg.index[propsAbove]) - ys[i+1]
        else:
            ys[i] = len(dxg.index[propsAbove])
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
