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
from tqdm import tqdm

from typing import Union

CLEAN_DATA_DIR: str = os.path.join("Data", "Laurence-Data")

from data_handler import DataHandler
from drug_search import make_dir

class CorrelationPlotter(DataHandler):
    def __init__():
        super().__init__((("AllByAll", os.path.join(CLEAN_DATA_DIR, "AllGenesByAllDrugs.tsv"))))
    
    def plot_drug_correlations(self) -> None:
        gxd = self.datasets["AllByAll"]
        sns.set_theme()
        results_dir = os.path.join("Data", "Results", "Drug-gene correlation frequency histograms")
        make_dir(results_dir)
        # Go through rows (i.e. drugs) for drug score distribution analysis
        print("Analysing drug correlation score distributions...")
        for index, row in tqdm(gxd.iterrows(), total = gxd.shape[0]):
            drug, scores = row["Drug"], row.values[1:]
            self.save_histogram(scores, f"{drug}-gene LOG correlations", results_dir)
    
    def plot_gene_correlations(self) -> None:
        gxd = self.datasets["AllByAll"]
        sns.set_theme()
        # Go through columns - gene score distribution analysis
        results_dir = os.path.join("Data", "Results", "Gene-drug correlation frequency histograms")
        make_dir(results_dir)
        print("Analysing gene correlation score distributions...")
        for gene in tqdm(gxd.columns[1:]):
            scores = gxd[gene].values
            self.save_histogram(scores, f"{gene}-drug LOG correlations", results_dir)
        return
    
    def save_histogram(self, scores: Union[np.ndarray, list, tuple], titlebase: str, results_dir: str) -> None:
        counts, bins = np.histogram(scores, bins = np.arange(-1, 1, 0.05))
        logcounts = np.log(counts)
        logcounts[logcounts == -np.inf] = 0
        for y, ylabel, title in zip((counts, logcounts), ("Frequency", "Log frequency"), \
                                            (titlebase.replace("LOG ",""), titlebase.replace("LOG", "log"))):
            plt.stairs(y, bins, fill = True)
            plt.plot([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)], y, color = "black")
            plt.xlabel("Survivability correlation")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.savefig(os.path.join(results_dir, title+" histogram.png"))
            plt.clf()