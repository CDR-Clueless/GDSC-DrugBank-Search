#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 19 Nov 2026

@author: jds40
"""

import numpy as np
from matplotlib import pyplot as plt

from file_imports import import_allbyall

THRESH: float = 1.645
CRITICAL_PROP: float = 0.8

def main():
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
