#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 5 Nov 2026

@author: jds40
"""

import os
import numpy as np

from matplotlib import pyplot as plt

from file_imports import import_allbyall

def main():
    # Import Data
    dxg = import_allbyall()
    # Get counts for number of genes >= 1.645 Z-Score per drug
    results = {}
    for drug in dxg.columns:
        arr = np.array(dxg[drug].values, dtype = float)
        arr = arr[~np.isnan(arr)]
        zs = np.divide(arr - np.mean(arr), np.std(arr))
        results[drug] = zs[zs >= 1.645].shape[0]
    xs, ys = list(results.keys()), list(results.values())
    xs = [x for _, x in sorted(zip(ys, xs))][::-1]
    ys = sorted(ys)[::-1]
    plt.bar(xs, ys)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.show()
    return

if(__name__=="__main__"):
    main()
