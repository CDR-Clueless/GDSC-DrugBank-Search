#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 15 May 2026

@author: jds40
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import json

GDSC_DEFAULT_PKI_FILE: str = os.path.join("Data", "Results", "pKi-AllDrugsByAllGenes.tsv")
GDSC_DEFAULT_IC50_FILE: str = os.path.join("Data", "Results", "LN_IC50-AllDrugsByAllGenes.tsv")
GDSCC_DEFAULT_IC50_FILE: str = os.path.join("Data", "Results", "Survivability-Correlations", "GDSCC-Single-LN_IC50-AllDrugsByAllGenes.tsv")
GDSCC_DEFAULT_EMAX_FILE: str = os.path.join("Data", "Results", "Survivability-Correlations", "GDSCC-Single-eMax-AllDrugsByAllGenes.tsv")


class CrossResponse:
    def __init__(self, response1_raw: str = "GDSC-pKi", response2_raw: str = "GDSC-IC50") -> None:
        """Import response1 and response2 data

        Args:
            response1 (str, optional): _description_. Defaults to GDSC pKi data.
            response2 (str, optional): _description_. Defaults to GDSC IC50 data.
        """
        # Translate input variables into data paths
        response1, response2 = response1_raw.lower().replace(" ",""), response2_raw.lower().replace(" ","")
        responseLocs = {"gdsc": {"pki": GDSC_DEFAULT_PKI_FILE, "ic50": GDSC_DEFAULT_IC50_FILE},
                       "gdscc": {"emax": GDSCC_DEFAULT_EMAX_FILE, "ic50": GDSCC_DEFAULT_IC50_FILE}}
        response1Loc = responseLocs[response1.split("-")[0]][response1.split("-")[-1]]
        response2Loc = responseLocs[response2.split("-")[0]][response2.split("-")[-1]]
        # Load in Data
        self.response1 = pd.read_csv(response1Loc, sep = "\t")
        self.response1.set_index("symbol", inplace=True)
        self.response2 = pd.read_csv(response2Loc, sep = "\t")
        self.response2.set_index("symbol", inplace=True)
        # Save data sources for graph titles
        self.response1Name: str = response1_raw.split("-")[-1]
        self.response2Name: str = response2_raw.split("-")[-1]
        if(response1_raw.split("-")[0].lower()!=response2_raw.split("-")[0].lower()):
            self.responseSource: str = response1_raw.split("-")[0]+"-"+response2_raw.split("-")[0]
        else:
            self.responseSource: str = response1_raw.split("-")[0]
        return
    
    def cross_compare(self, method: str = "pearson", saveDir: Optional[str] = None) -> None:
        """Compare the survivability correlations for each drug using pKi vs IC50

        Args:
            method (str, optional): _description_. Defaults to "pearson".
        """

        # Set save directory
        if(saveDir is None):
            saveDir = os.path.join("Data", "Results", f"{self.responseSource}-CrossAnalysis")
        
        results, method, correlations = {}, method.lower().strip(), []
        for drug in self.response1.columns:
            # If this drug isn't in the IC50 data, skip it
            if(drug not in self.response2.columns):
                continue
            # Fetch the relevant data
            response1, response2 = self.response1[drug].values, self.response2[drug].values
            # Remove NaN values
            r1nan, r2nan = np.isnan(response1), np.isnan(response2)
            rnan = np.logical_or(r1nan, r2nan)
            response1, response2 = response1[~rnan], response2[~rnan]
            # Ensure there are still some valid results to continue
            if(response1.shape[0]==0):
                results[drug] = {"Pearson result": float("NaN"), "Pearson p-value": float("NaN")}
                continue
            # Run the analysis
            if(method=="pearson"):
                pr, pp = pearsonr(response1, response2)
                results[drug] = {"Pearson result": pr, "Pearson p-value": pp}
                correlations.append(pr)
        # Save result data
        with open(saveDir+".json", "w") as f:
            json.dump(results, f, indent = 4)
        # Save histogram of results
        counts, bins = np.histogram(correlations, np.linspace(-1, 1, 101, endpoint=True))
        plt.stairs(counts, bins)
        plt.title(f"{self.responseSource} {self.response1Name}-{self.response2Name} response drug correlations")
        plt.xlabel(f"{method.capitalize()} Correlation")
        plt.ylabel("Count")
        plt.savefig(saveDir+".png")
        plt.clf()
        plt.close()
        return