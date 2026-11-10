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
    results, resultsAbs, geneCounts = {}, {}, {}
    for drug in dxg.columns:
        arr = np.array(dxg[drug].values, dtype = float)
        arr = arr[~np.isnan(arr)]
        zs = np.divide(arr - np.mean(arr), np.std(arr))
        zsAbs = np.divide(abs(arr) - np.mean(abs(arr)), np.std(abs(arr)))
        #results[drug] = zs[zs >= 1.645].shape[0]
        results[drug] = np.sum(zs >= 1.645)
        #resultsAbs[drug] = zsAbs[zsAbs >= 1.645].shape[0]
        resultsAbs[drug] = np.sum(zsAbs >= 1.645)
        geneCounts[drug] = zs.shape[0]
    xs, ys = list(results.keys()), list(results.values())
    xsRaw, totsRaw = [x for _, x in sorted(zip(ys, xs))][::-1], [tot for _, tot in sorted(zip(ys, list(geneCounts.values())))][::-1]
    ys = sorted(ys)[::-1]
    props = [np.divide(ys[i], totsRaw[i]) for i in range(len(ys))]
    fig, ax = plt.subplots(figsize = (19.2, 12.8))
    ax.bar(range(len(xsRaw)), ys, width = 1.0, color = "blue")
    axProp = ax.twinx()
    axProp.bar(range(len(xsRaw)), props, width = 1.0, color = "blue")
    ax.set_ylabel("Number of genes p < 0.05")
    axProp.set_ylabel("Proportion of genes p < 0.05")
    ax.set_ylim([min(ys)*0.99, max(ys)*1.01])
    axProp.set_ylim([min(props)*0.99, max(props)*1.01])
    ax.grid(visible = True, which = "major", axis = "y")
    ax.set_zorder(2)
    axProp.set_zorder(1)
    #plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.set_title("Number of genes with Z >= 1.645 i.e. p < 0.05")
    plt.show()
    fig, ax = plt.subplots(figsize = (19.2, 12.8))
    ysAbs = list(resultsAbs.values())
    xsAbs, totsAbs = [x for _, x in sorted(zip(ysAbs, xs))][::-1], [tot for _, tot in sorted(zip(ysAbs, list(geneCounts.values())))][::-1]
    ysAbs = sorted(ysAbs)[::-1]
    propsAbs = [np.divide(ysAbs[i], totsAbs[i]) for i in range(len(ysAbs))]
    ax.plot(range(len(xs)), ys, color = "blue", label = "Standard SC")
    axProp = ax.twinx()
    axProp.plot(range(len(xs)), props, color = "blue")
    ax.plot(range(len(xsAbs)), ysAbs, color = "orange", label = "Absolute SC")
    axProp.plot(range(len(xsAbs)), propsAbs, color = "orange")
    ax.set_ylabel("Number of genes with p < 0.05")
    axProp.set_ylabel("Proportion of genes with p < .0.05")
    ax.grid(visible=True, which = "major", axis = "y")
    ax.set_zorder(1)
    axProp.set_zorder(0)
    ax.set_title("Comparison of Gene Z-Value Distributions using (Non-)Absolute SC values")
    ax.legend(loc = "upper right")
    plt.show()

    return

if(__name__=="__main__"):
    main()
