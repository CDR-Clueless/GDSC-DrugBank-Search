#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 15 May 2026

@author: jds40
"""

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import json

DEFAULT_PKI_FILE: str = os.path.join("Data", "Results", "pKi-AllDrugsByAllGenes.tsv")
DEFAULT_IC50_FILE: str = os.path.join("Data", "Results", "LN_IC50-AllDrugsByAllGenes.tsv")

class GDSC_CrossResponse:
    def __init__(self, pki: str = DEFAULT_PKI_FILE, ic50: str = DEFAULT_IC50_FILE) -> None:
        """Import pKi and IC50 data

        Args:
            pki (str, optional): _description_. Defaults to DEFAULT_PKI_FILE.
            ic50 (str, optional): _description_. Defaults to DEFAULT_IC50_FILE.
        """
        self.pki = pd.read_csv(pki, sep = "\t")
        self.pki.set_index("symbol", inplace=True)
        self.ic50 = pd.read_csv(ic50, sep = "\t")
        self.ic50.set_index("symbol", inplace=True)
        return
    
    def cross_compare(self, method: str = "pearson", saveDir = os.path.join("Data", "Results", "GDSC-CrossAnalysis")) -> None:
        """Compare the survivability correlations for each drug using pKi vs IC50

        Args:
            method (str, optional): _description_. Defaults to "pearson".
        """
        results, method, correlations = {}, method.lower().strip(), []
        for drug in self.pki.columns:
            # If this drug isn't in the IC50 data, skip it
            if(drug not in self.ic50.columns):
                continue
            # Fetch the relevant data
            pki, ic50 = self.pki[drug].values, self.ic50[drug].values
            # Run the analysis
            if(method=="pearson"):
                pr, pp = pearsonr(pki, ic50)
                results[drug] = {"Pearson result": pr, "Pearson p-value": pp}
                correlations.append(pr)
        # Save result data
        with open(saveDir+".json", "w") as f:
            json.dump(results, f, indent = 4)
        # Save histogram of results
        counts, bins = np.histogram(correlations, np.linspace(-1, 1, 41, endpoint=True))
        plt.stairs(counts, bins)
        plt.title("GDSC pKi-IC50 response drug correlations")
        plt.xlabel(f"{method.capitalize()} Correlation")
        plt.ylabel("Count")
        plt.savefig(saveDir+".png")
        plt.clf()
        plt.close()
        return