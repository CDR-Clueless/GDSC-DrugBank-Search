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
import json

from typing import Union
from typing import Optional

CLEAN_DATA_DIR: str = os.path.join("Data", "Laurence-Data")

from data_handler import DataHandler
from drug_search import make_dir

class CorrelationPlotter(DataHandler):
    def __init__(self):
        # Call super init function and insert the relevant AllGenesByAllDrugs data into the instance
        super().__init__((("AllByAll", os.path.join(CLEAN_DATA_DIR, "AllGenesByAllDrugs.tsv"))))
        self.datasets["AllByAll"] = self.datasets["AllByAll"].rename(columns = {"Unnamed: 0": "Drug"})
    
    def plot_all(self, stds:list = [1, 2, 3], quantiles: list = []) -> None:
        self.plot_drug_correlations(stds, quantiles)
        self.plot_gene_correlations(stds, quantiles)
        self.plot_sd_cumulative()
        return
    
    def __get_stats_markers(self, stds: list = [], quantiles: list = []) -> dict[str, list[float]]:
        # Set up dictionary output for quantiles and SDs json
        resVals = {"quantiles": [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999], "devs": [1., 2., 3.]}
        for q in quantiles:
            if q not in resVals["quantiles"]:
                resVals["quantiles"].append(q)
        for sd in stds:
            if(sd not in resVals["devs"]):
                resVals["devs"].append(sd)
                resVals["devs"].append(-1*sd)
        resVals["quantiles"] = sorted(resVals["quantiles"])
        resVals["devs"] = sorted(resVals["devs"])
        return resVals
    
    def __get_within_sds(self, vals: Union[list, tuple, np.ndarray], stds: list = [1., 2., 3.]) -> dict[float|str, int]:
        # First, remove duplicates within stds and sort the remainder, set up a results dictionary,
        # and convert the input to a numpy array for more efficient calculations
        stds, res, vals = list(set(sorted([abs(s) for s in stds]))), {}, np.array(vals)
        for sd in stds:
            sdRange = (np.mean(vals)+(np.std(vals)*sd), np.mean(vals)-(np.std(vals)*sd))
            # Get a a filtered array consisting of only the values which fall in the desired range
            filtered = vals[(vals < sdRange[0]) & (vals > sdRange[1])]
            res[sd] = len(filtered)
        # Add the total number of samples to the dictionary
        res["Total"] = len(vals)
        return res

    def plot_drug_correlations(self, stds: list = [], quantiles: list = []) -> None:
        gxd = self.datasets["AllByAll"]
        sns.set_theme()
        results_dir = os.path.join("Data", "Results", "Drug-gene correlation frequency histograms")
        make_dir(results_dir)
        # Set up dictionary output for quantiles and SDs json
        sres, resVals = {}, self.__get_stats_markers(stds, quantiles)
        # Go through rows (i.e. drugs) for drug score distribution analysis
        for index, row in tqdm(gxd.iterrows(), total = gxd.shape[0], desc = "Analysing drug correlation score distributions"):
            drug, scores = row["Drug"], np.array(row.values[1:], dtype = float)
            # Remove NaN values
            scores = scores[~np.isnan(scores)]
            self.save_histogram(scores, f"{drug}-gene LOG correlations", results_dir, stds, quantiles)
            sres[drug] = {"standard deviations": \
                            {d: np.mean(scores)+(np.std(scores)*d) for d in resVals["devs"]},
                          "quantiles": \
                            {q: np.quantile(scores, q) for q in resVals["quantiles"]},
                          "standard deviation counts": \
                            self.__get_within_sds(scores, resVals["devs"])}
        with open(os.path.join(results_dir, "stats.json"), "w") as f:
            json.dump(sres, f, indent = 4)
        return
    
    def plot_gene_correlations(self, stds: list = [], quantiles: list = []) -> None:
        gxd = self.datasets["AllByAll"]
        sns.set_theme()
        # Go through columns - gene score distribution analysis
        results_dir = os.path.join("Data", "Results", "Gene-drug correlation frequency histograms")
        make_dir(results_dir)
        # Set up dictionary output for quantiles json
        sres, resVals = {}, self.__get_stats_markers(stds, quantiles)
        for gene in tqdm(gxd.columns[1:], desc = "Analysing gene correlation score distributions"):
            scores = np.array(gxd[gene].values, dtype = float)
            # Remove NaN values
            scores = scores[~np.isnan(scores)]
            self.save_histogram(scores, f"{gene}-drug LOG correlations", results_dir, stds, quantiles)
            sres[gene] = {"standard deviations": \
                            {d: np.mean(scores)+(np.std(scores)*d) for d in (resVals["devs"] + [-1*sdm for sdm in resVals["devs"]])},
                          "quantiles": \
                            {q: np.quantile(scores, q) for q in resVals["quantiles"]},
                          "standard deviation counts": \
                            self.__get_within_sds(scores, resVals["devs"])}
        with open(os.path.join(results_dir, "stats.json"), "w") as f:
            json.dump(sres, f, indent = 4)
        return
    
    def save_histogram(self, scores: Union[np.ndarray, list, tuple], titlebase: str, results_dir: str,
                       stds: list = [], quantiles: list = []) -> None:
        # Get counts and bins for histograms
        counts, bins = np.histogram(scores, bins = np.arange(-1, 1, 0.05))
        # Get the (corrected) log of these counts for the log graph
        logcounts = np.log(counts)
        logcounts[logcounts == -np.inf] = 0
        # Loop through relevant variables to produce regular and logged histograms
        for y, ylabel, title in zip((counts, logcounts), ("Frequency", "Log frequency"), \
                                            (titlebase.replace("LOG ",""), titlebase.replace("LOG", "log"))):
            # Main histogram
            plt.stairs(y, bins, fill = True)
            # Black bars separating bins
            plt.plot([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)], y, color = "black")
            plt.xlabel("Survivability correlation")
            plt.ylabel(ylabel)
            plt.title(title)
            # Add standard deviation bars if appropriate
            if(len(stds)>0):
                avg, dev = np.mean(scores), np.std(scores)
                for std in stds:
                    plt.plot([avg-(dev*std)]*2, [0, max(y)*1.01], linestyle = "--", color = "g")
                    plt.plot([avg+(dev*std)]*2, [0, max(y)*1.01], linestyle = "--", color = "g")
                    plt.text(avg-(dev*std), max(y)*1.05, f"-{std} SDs", size = "xx-small")
                    plt.text(avg+(dev*std), max(y)*1.05, f"+{std} SDs", size = "xx-small")
            # Add quantile markings if appropriate
            if(len(quantiles)>0):
                vals = np.quantile(scores, quantiles)
                for quantile, val, x in zip(quantiles, vals, np.linspace(-1, 1, len(quantiles), endpoint=False)):
                    plt.plot([val]*2, [0, max(y)*1.01], linestyle = "--", color = "r")
                    plt.text(x, max(y)*1.05, f"Quantile {quantile} = {round(val, 3)}", size = "xx-small")
            plt.savefig(os.path.join(results_dir, title+" histogram.png"))
            plt.clf()
        return
    
    def plot_sd_cumulative(self, mode: str = "drug", stds: list = [1.0, 2.0, 3.0]):
        # If the mode is 'both', call this function with the 2 sub-options
        if(mode.lower()=="both"):
            for m in ["drug", "gene"]:
                self.plot_sd_cumulative(mode = m, stds = stds)
            return
        # Set seaborn theme for better graphs
        sns.set_theme()
        # Get the relevant directory
        ddir = os.path.join("Data", "Results")
        if(mode.lower()=="gene"):
            data_dir = os.path.join(ddir,  "Gene-drug correlation frequency histograms", "stats.json")
        else:
            data_dir = os.path.join(ddir, "Drug-gene correlation frequency histograms", "stats.json")
        if(os.path.exists(data_dir)==False):
            print(f"No results found at {data_dir}; please (re-)run plot_all, or plot_(drug/gene)_correlations if only one output is desired")
            return
        with open(data_dir, "r") as f:
            data = json.load(f)
        # Check the requested standard deviations from mean are available
        check = list(data[list(data.keys())[0]]["standard deviations"].keys())
        if(type(check[0]) == str):
            savedStr = True
            check = [float(key) for key in check]
        else:
            savedStr = False
        for i in range(len(stds))[::-1]:
            if(stds[i] not in check):
                print(f"Standard deviation value {stds[i]} not found in results - skipping (rerun histogram plotting with desired values)")
                stds.pop(i)
        # Get the relevant standard deviation values into a more usable format
        devs = {sd: [] for sd in stds}
        # NOTE: The if-else check is outside the for loops because even though it means the code looks bigger/is repetitive,
        # it means this check is only performed once wather than in O(dg*sd) time
        if(savedStr):
            # dg represents that this could be drugs or genes we're iterating through
            for dg in tqdm(data, desc = "re-organising data"):
                rel = data[dg]["standard deviations"]
                for sd in stds:
                    fKey = str(sd)
                    devs[sd].append(rel[fKey])
        else:
            for dg in tqdm(data, desc = "re-organising data"):
                rel = data[dg]["standard deviations"]
                for sd in stds:
                    fKey = sd
                    devs[sd].append(rel[fKey])
        for sd in devs:
            # Get the deviation values and appropriate x values to show this as a cumulative frequency plot
            # Note: 0 is added so all graph lines start from (0, 0) rather than (lowest correlation value, 0)
            sdVals = np.array([0]+devs[sd], dtype=float)
            sdVals = np.sort(sdVals)
            ys = np.array(range(sdVals.shape[0]))/float(sdVals.shape[0]-1)
            # Plot this as a CDF
            plt.plot(sdVals, ys, label = f"{sd} SDs")
        # Labels and titles
        plt.xlabel("Survivability correlation")
        plt.ylabel("Cumulative frequency")
        plt.title("Cumulative frequency of Standard deviation boundaries")
        plt.legend()
        plt.savefig(os.path.join(ddir, f"All {mode.capitalize()}s survivability correlation standard deviation CDF.png"))
        plt.clf()
        plt.close()
        return