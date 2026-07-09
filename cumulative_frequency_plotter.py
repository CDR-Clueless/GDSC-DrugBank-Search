#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 9 Jul 2026

@author: jds40
"""

import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

def main():
    df = import_data()
    minimums, maximums = [], []
    for drug in df.columns:
        rel = df[drug].values
        minimums.append(np.nanmin(rel))
        maximums.append(np.nanmax(rel))
    minimums, maximums = np.array(sorted(minimums)), np.array(sorted(maximums))
    xsmin = np.array(range(minimums.shape[0]))/float(minimums.shape[0]-1)
    xsmax = np.array(range(maximums.shape[0]))/float(maximums.shape[0]-1)
    plt.plot(xsmin, minimums, label = "Minimum SC Value")
    plt.plot(xsmax, maximums, label = "Maximum SC Value")
    plt.legend()
    plt.xlabel("Cumulative Frequency")
    plt.ylabel("Survivability Correlation")
    plt.show()
    return

def import_data():
    data = pd.read_csv(os.path.join("Data", "Results", "Survivability-Correlations", "pIC50-AllDrugsByAllGenes.tsv"), sep = "\t")
    data.set_index("symbol", inplace=True)
    return data


if(__name__=="__main__"):
    main()