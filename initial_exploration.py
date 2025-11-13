#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2025

@author: jds40
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from copy import deepcopy
import json
from tqdm import tqdm
import ast

from typing import Optional

from searcher import Searcher

CLEAN_DATA_DIR: str = os.path.join("Data", "Laurence-Data")

def main():
    gxd = pd.read_csv(os.path.join(CLEAN_DATA_DIR, "AllGenesByAllDrugs.tsv"), sep = "\t")
    gxd = gxd.rename(columns = {"Unnamed: 0": "Drug"})
    sns.set_theme()
    results_dir = os.path.join("Data", "Results", "Drug-gene correlation frequency histograms")
    make_dir(results_dir)
    # Go through rows - drug score distribution analysis
    print("Analysing drug correlation score distributions...")
    for index, row in tqdm(gxd.iterrows(), total = gxd.shape[0]):
        drug, scores = row["Drug"], row.values[1:]
        counts, bins = np.histogram(scores, bins = np.arange(-1, 1, 0.05))
        logcounts = np.log(counts)
        logcounts[logcounts == -np.inf] = 0
        for y, ylabel, filename in zip((counts, logcounts), ("Frequency", "Log frequency"), \
                                              (f"{drug}-gene correlation histogram.png", f"{drug}-gene log correlation histogram.png")):
            plt.stairs(y, bins, fill = True)
            plt.plot([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)], y, color = "black")
            plt.xlabel("Survivability correlation")
            plt.ylabel(ylabel)
            plt.title(f"{drug} drug-gene survivability correlations")
            plt.savefig(os.path.join(results_dir, filename))
            plt.clf()
    # Go through columns - gene score distribution analysis
    results_dir = os.path.join("Data", "Results", "Gene-drug correlation frequency histograms")
    make_dir(results_dir)
    print("Analysing gene correlation score distributions...")
    for gene in tqdm(gxd.columns[1:]):
        scores = gxd[gene].values
        counts, bins = np.histogram(scores, bins = np.arange(-1, 1, 0.05))
        logcounts = np.log(counts)
        logcounts[logcounts == -np.inf] = 0
        for y, ylabel, filename in zip((counts, logcounts), ("Frequency", "Log frequency"), \
                                              (f"{gene}-drug correlation histogram.png", f"{gene}-drug log correlation histogram.png")):
            plt.stairs(y, bins, fill = True)
            plt.plot([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)], y, color = "black")
            plt.xlabel("Survivability correlation")
            plt.ylabel(ylabel)
            plt.title(f"{gene} gene-drug survivability correlations")
            plt.savefig(os.path.join(results_dir, filename))
            plt.clf()
    return

if __name__ == "__main__":
    main()
