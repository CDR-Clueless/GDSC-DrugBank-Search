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
import diptest
from copy import deepcopy
from scipy.optimize import curve_fit
import multiprocessing as mp

from typing import Union
from typing import Optional

CLEAN_DATA_DIR: str = os.path.join("Data", "Laurence-Data")

from data_handler import DataHandler
from drug_search import make_dir

class CorrelationPlotter(DataHandler):
    def __init__(self, coreCount: Optional[int]):
        # Call super init function and insert the relevant AllGenesByAllDrugs data into the instance
        super().__init__((("AllByAll", os.path.join(CLEAN_DATA_DIR, "AllGenesByAllDrugs.tsv"))))
        self.datasets["AllByAll"] = self.datasets["AllByAll"].rename(columns = {"Unnamed: 0": "Drug"})
        # Define number of available cores to utilise in multiprocessing
        if(coreCount is int):
            if(coreCount>0):
                self.coreCount = coreCount
            else:
                self.coreCount = 1
        else:
            self.coreCount = max(mp.cpu_count()-2, 1)
    
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
    
    def __get_modality_entry(self, dist: Union[list, tuple, np.ndarray]) -> dict:
        res = diptest.diptest(dist)
        if(res[1]<0.15):
            modality = "bimodal"
        elif(res[1]>0.4):
            modality = "unimodal"
        else:
            modality = "unclear"
        return {"modality": modality, "HDS": res[0], "HDS p-value": res[1], "mean": np.mean(dist)}

    def plot_diptest_histogram(self, mode: str = "drug", outdir: str = os.path.join("Data", "Results")):
        data = self.datasets["AllByAll"]
        if(mode.lower()=="both"):
            for m in ["drug", "gene"]:
                self.plot_diptest_histogram(m, outdir)
            return
        if(mode.lower()=="drug"):
            results = [diptest.diptest(data.iloc[i][1:].values) for i in range(len(data))]
        elif(mode.lower()=="gene"):
            results = [diptest.diptest(data[col].values) for col in data.columns[1:]]
        else:
            print(f"Mode '{mode}' not recognised. Please use 'drug', 'gene' or 'both' as the mode")
            return
        pvals = [t[1] for t in results]
        counts, bins = np.histogram(pvals, bins = np.arange(0.0, 1.05, 0.05))
        plt.stairs(counts, bins)
        plt.xlabel("p-value")
        plt.ylabel("Distribution frequency")
        if(mode=="drug"):
            plt.title("Results of Hartigan's dip statistic test for drug-gene survivability correlations")
            plt.savefig(os.path.join(outdir, "HDS drug-gene histogram.png"))
        else:
            plt.title("Results of Hartigan's dip statistic test for gene-drug survivability correlations")
            plt.savefig(os.path.join(outdir, "HDS gene-drug histogram.png"))
        plt.clf()
        plt.close()
        return

    def plot_drug_correlations(self, stds: list = [], quantiles: list = []) -> None:
        gxd = self.datasets["AllByAll"]
        sns.set_theme()
        results_dir = os.path.join("Data", "Results", "Drug-gene correlation frequency histograms")
        make_dir(results_dir)
        # Set up dictionary output for quantiles and SDs json
        sres, resVals = {}, self.__get_stats_markers(stds, quantiles)
        # Go through rows (i.e. drugs) for drug score distribution analysis in parallel
        with mp.Pool(self.coreCount) as p:
            results = p.starmap(self.drug_correlations_worker,
                                [(row, results_dir, stds, quantiles, resVals) for i, row in gxd.iterrows()])
        # Add all these parallel-calculated results to main results output
        for result in results:
            sres.update(result)
        with open(os.path.join(results_dir, "stats.json"), "w") as f:
            json.dump(sres, f, indent = 4)
        return

    def drug_correlations_worker(self, row: pd.Series, results_dir: str, stds: list, quantiles: list, resVals: dict) -> dict:
        drug, scores = row["Drug"], np.array(row.values[1:], dtype = float)
        # Remove NaN values
        scores = scores[~np.isnan(scores)]
        extra_data = self.save_histogram(scores, f"{drug}-gene LOG correlations", results_dir, stds, quantiles)
        newdict = {"standard deviations": \
                        {d: np.mean(scores)+(np.std(scores)*d) for d in resVals["devs"]},
                        "quantiles": \
                        {q: np.quantile(scores, q) for q in resVals["quantiles"]},
                        "standard deviation counts": \
                        self.__get_within_sds(scores, resVals["devs"])}
        newdict.update(extra_data)
        return {drug: newdict}
    
    def plot_gene_correlations(self, stds: list = [], quantiles: list = []) -> None:
        gxd = self.datasets["AllByAll"]
        sns.set_theme()
        # Go through columns - gene score distribution analysis
        results_dir = os.path.join("Data", "Results", "Gene-drug correlation frequency histograms")
        make_dir(results_dir)
        # Set up dictionary output for quantiles json
        sres, resVals = {}, self.__get_stats_markers(stds, quantiles)
        with mp.Pool(self.coreCount) as p:
            results = p.starmap(self.gene_correlations_worker,
                                [(gene, gxd[gene].values, results_dir, stds, quantiles, resVals) for gene in gxd.columns[1:]])
        # Add all these parallel-calculated results to main results output
        for result in results:
            sres.update(result)
        with open(os.path.join(results_dir, "stats.json"), "w") as f:
            json.dump(sres, f, indent = 4)
        return
    
    def gene_correlations_worker(self, gene, values, results_dir, stds, quantiles, resVals) -> dict:
        scores = np.array(values, dtype = float)
        # Remove NaN values
        scores = scores[~np.isnan(scores)]
        extra_data = self.save_histogram(scores, f"{gene}-drug LOG correlations", results_dir, stds, quantiles)
        newdict = {"standard deviations": \
                        {d: np.mean(scores)+(np.std(scores)*d) for d in (resVals["devs"] + [-1*sdm for sdm in resVals["devs"]])},
                        "quantiles": \
                        {q: np.quantile(scores, q) for q in resVals["quantiles"]},
                        "standard deviation counts": \
                        self.__get_within_sds(scores, resVals["devs"])}
        newdict.update(extra_data)
        return {gene: newdict}
    
    def save_histogram(self, scores: Union[np.ndarray, list, tuple], titlebase: str, results_dir: str,
                       stds: list = [], quantiles: list = [], plot_curve: bool = True,
                       test_modality: bool = True) -> dict:
        # Set up output dictionary if necessary
        output = {}
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
            # Black line showing trend
            xs = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
            plt.plot(xs, y, color = "black")
            plt.xlabel("Survivability correlation")
            plt.ylabel(ylabel)
            plt.title(title)
            # Add standard deviation bars if appropriate
            if(len(stds)>0):
                avg, dev = np.mean(scores), np.std(scores)
                if("log" in title.lower()):
                    avg, dev = np.log(avg), np.log(dev)
                for std in stds:
                    plt.plot([avg-(dev*std)]*2, [0, max(y)*1.01], linestyle = "--", color = "g")
                    plt.plot([avg+(dev*std)]*2, [0, max(y)*1.01], linestyle = "--", color = "g")
                    plt.text(avg-(dev*std), max(y)*1.05, f"-{std} SDs", size = "xx-small")
                    plt.text(avg+(dev*std), max(y)*1.05, f"+{std} SDs", size = "xx-small")
            # Add quantile markings if appropriate
            if(len(quantiles)>0):
                vals = np.quantile(scores, quantiles)
                if("log" in title.lower()):
                    vals = [np.log(val) for val in vals]
                for quantile, val, x in zip(quantiles, vals, np.linspace(-1, 1, len(quantiles), endpoint=False)):
                    plt.plot([val]*2, [0, max(y)*1.01], linestyle = "--", color = "r")
                    plt.text(x, max(y)*1.05, f"Quantile {quantile} = {round(val, 3)}", size = "xx-small")
            # Get modality details if appropriate
            if((test_modality or plot_curve) and "log" not in title.lower()):
                output["modality details"] = self.__get_modality_entry(scores)
                curve_type = output["modality details"]["modality"]
            # Add curve fit if appropriate
            if(plot_curve and "log" not in title.lower()):
                #try:
                func = {"gaussian": gaussian, "bimodal": bimodal, "unimodal": gaussian, "unclear": gaussian}[curve_type]
                ig = curve_guess(xs, y, curve_type)
                try:
                    params, cov = curve_fit(func,
                                            xdata = xs, 
                                            ydata = y,
                                            p0 = ig, maxfev = 25000)
                    curve_xs = np.linspace(xs[0], xs[-1], len(xs)*100)
                    plt.plot(curve_xs, func(curve_xs, *params), color = "g", label = "fitted curve")
                    # Save parameters if this isn't the log graph
                    if("log" not in title.lower()):
                        paramdict, covdict = {}, {}
                        covarr = np.sqrt(np.diag(cov))
                        for i in range(int(len(params)/3)):
                            for param, j in zip(("A", "mu", "sigma"), (0, 1, 2)):
                                paramdict[f"{param}{i}"] = params[j+(3*i)]
                                covdict[f"{param}{i}"] = covarr[j+(3*i)]
                        output["curve parameters"] = deepcopy(paramdict)
                        output["curve errors standard deviation"] =  deepcopy(covdict)
                except Exception as e:
                    print(f"Error with {title}: {e}")
                    paramdict, covdict = {}, {}
                    for i in range({"gaussian": 1, "unimodal": 1, "unclear": 1, "bimodal": 2}[curve_type]):
                        for param, j in zip(("A", "mu", "sigma"), (0, 1, 2)):
                            paramdict[f"{param}{i}"] = "NaN"
                            covdict[f"{param}{i}"] = "NaN"
                    output["curve parameters"] = deepcopy(paramdict)
                    output["curve errors standard deviation"] =  deepcopy(covdict)
            plt.legend()
            plt.savefig(os.path.join(results_dir, title+" histogram.png"))
            plt.clf()
        return output
    
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

def curve_guess(xs: Union[np.ndarray, list, tuple], ys: Union[np.ndarray, list, tuple], mode: str = "gaussian", sigma = 0.05) -> tuple:
    """
    
    Find initial guess for gaussian/bimodal gaussian parameters as the input for a curve optimisation

    Args:
        xs (Union[np.ndarray, list, tuple]): List of x values
        ys (Union[np.ndarray, list, tuple]): List of y values
        mode (str, optional): Type of curve being guessed at. Defaults to "gaussian"; accepted values are "gaussian", "unimodal", and "bimodal".
        sigma (float, optional): Sigma value to be used, as this is not guessed at. Defaults to 0.05.

    Returns:
        tuple: Tuple of parameters (A, mu, sigma); can extend to length of 6, 9... if the mode uses 2 gaussians (bimodal), 3 etc.
    """
    if(mode=="gaussian" or mode == "unimodal"):
        peak = get_peaks(xs, ys, limit = 1)[0]
        return np.array([peak[1], peak[0], sigma], dtype = float)
    elif(mode=="bimodal"):
        peaks = get_peaks(xs, ys, limit = 2)
        # If only a single peak was found, guesstimate the other entry
        if(len(peaks)<2):
            peaks.append((max(ys)/4, max(xs)*0.7))
        return np.array([peaks[0][1], peaks[0][0], sigma, peaks[1][1], peaks[1][0], sigma], dtype = float)

def get_peaks(xs: Union[np.ndarray, list, tuple], ys: Union[np.ndarray, list, tuple], limit: int = 1) -> list:
    """
    
    Estimate the peaks in a data series with x and y values

    Args:
        xs (Union[np.ndarray, list, tuple]): Array of x values
        ys (Union[np.ndarray, list, tuple]): Array of y values
        limit (int, optional): Limit on the number of peaks found and returned by the function. Defaults to 1.

    Returns:
        list: A list of peaks in the format [(x(peak1), y(peak1)), (x(peak2), y(peak2)), ...]
    """
    # Get a list of tuples (x(i), y(i)) where y(i) is greater than both y(i-1) and y(i+1)
    peaks = [(xs[i], ys[i]) for i in range(1, len(xs)-1) if (ys[i-1]<=ys[i] and ys[i+1]<=ys[i])]
    # Remove extra peaks
    if(len(peaks)>limit):
        refined = [(0, -np.inf) for _ in range(limit)]
        for tup in peaks:
            i: int = 0
            while(i<len(refined)):
                if(tup[1]>refined[i][1]):
                    # First, shift everything down 1 in the refined list
                    for j in range(i+1, len(refined))[::-1]:
                        refined[j] = deepcopy(refined[j-1])
                    # Next, insert the new peak into the refined list and stop searching where the refined list entries can be replaced
                    refined[i] = deepcopy(tup)
                    break
                i += 1
        return refined
    # Return peaks found
    return peaks

def gaussian(x, A, mu, sigma):
    return A*np.exp(-np.divide(np.power(x-mu, 2),(2*np.power(sigma, 2))))

def bimodal(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2)