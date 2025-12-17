#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 11 Dec 2025

@author: jds40
"""

import os
import math
import numpy as np
from matplotlib import pyplot as plt
from typing import Union, Optional
from copy import deepcopy

from data_handler import DataHandler
CLEAN_DATA_DIR: str = os.path.join("Data", "Laurence-Data")

class ModalityAnalyzer(DataHandler):
    def __init__(self, datasets: Union[list, tuple, None] = None):
        super().__init__(datasets)
        if("drug modality summary" not in self.datasets.keys()):
            self.load_data("drug modality summary", os.path.join("Data", "Results", "Drug-gene correlation frequency histograms", "stats.json"))
        if("gene modality summary" not in self.datasets.keys()):
            self.load_data("gene modality summary", os.path.join("Data", "Results", "Gene-drug correlation frequency histograms", "stats.json"))
        if("dxg" not in self.datasets.keys()):
            self.load_data("AllByAll", os.path.join(CLEAN_DATA_DIR, "AllGenesByAllDrugs.tsv"))
            self.datasets["AllByAll"] = self.datasets["AllByAll"].rename(columns = {"Unnamed: 0": "Drug"})
            self.datasets["AllByAll"] = self.datasets["AllByAll"].set_index("Drug")
        # Sort internal datasets into unique modalities
        self.__sort_modalities()
        return
    
    # Sort internal dataset into unimodal, unclear and bimodal dictionaries
    def __sort_modalities(self):
        for internalname in ["drug modality summary", "gene modality summary"]:
            uni, unc, bim = {}, {}, {}
            for entry in self.datasets[internalname]:
                deets = self.datasets[internalname][entry]["modality details"]["modality"]
                if(deets=="unimodal"):
                    uni[entry] = deepcopy(self.datasets[internalname][entry])
                elif(deets=="unclear"):
                    unc[entry] = deepcopy(self.datasets[internalname][entry])
                elif(deets=="bimodal"):
                    bim[entry] = deepcopy(self.datasets[internalname][entry])
                else:
                    print(f"Could not sort unrecognised modality: {deets}")
            # Overwrite entry
            self.datasets[internalname] = {"unimodal": deepcopy(uni), "bimodal": deepcopy(bim), "unclear": deepcopy(unc)}
        return
    
    # Plot cumulative frequency graphs for each modality type, plotting medians and threshold values
    def plot_cf(self, mode: str = "drug", save_dir = os.path.join("Data", "Results", "modality graphs"),
                keep_unclear: bool = False, overlay_histogram: bool = True, hist_width: float = 0.025):
        mode = mode.lower().strip()
        if(mode=="both"):
            self.plot_cf("drug", save_dir, keep_unclear, overlay_histogram)
            self.plot_cf("gene", save_dir, keep_unclear, overlay_histogram)
            return
        
        # Get a dictionary containing all the drug/gene survivability values
        if(mode=="drug"):
            survivability_arrays = {drug: self.datasets["AllByAll"].loc[[drug]].values for drug in self.datasets["AllByAll"].index}
        elif(mode=="gene"):
            survivability_arrays = {gene: self.datasets["AllByAll"][gene].values for gene in self.datasets["AllByAll"].columns}
        else:
            print(f"Unrecognised mode type: {mode}. Please use 'drug' or 'gene' as the mode.")
            return

        # Create dictionaries for results of medians and strong-correlation thresholds for different modality types
        meds, thresh = {"unimodal": [], "bimodal": [], "unclear": []}, {"unimodal": [], "bimodal": [], "unclear": []}

        # Get relevant data
        if(mode=="drug"):
            data = self.datasets["drug modality summary"]
        else:
            data = self.datasets["gene modality summary"]
        
        # Remove 'unclear' as an option if desired
        if(not keep_unclear):
            del meds["unclear"]
            del thresh["unclear"]
            del data["unclear"]

        # Insert relevant data into dictionaries - the median and threshold values
        for mtype in data:
            # dg stands for drug-gene, as depending on the function mode this variable could be iterating over drugs or genes
            for dg in data[mtype].keys():
                meds[mtype].append(float(data[mtype][dg]["quantiles"]["0.5"]))
                thresh[mtype].append(get_survivability_threshold(dg, data[mtype], survivability_arrays[dg]))
        
        # Get rid of any NaN values
        for mtype in data:
            for i in range(len(meds[mtype]))[::-1]:
                if(meds[mtype][i]!=meds[mtype][i]):
                    meds[mtype].pop(i)
            for i in range(len(thresh[mtype]))[::-1]:
                if(thresh[mtype][i]!=thresh[mtype][i]):
                    thresh[mtype].pop(i)
        
        # Format the values and into x and y cumulative frequency values
        ysm, yst = {}, {}
        for key in meds:
            meds[key] = np.array([0] + sorted(meds[key]), dtype = float)
            thresh[key] = np.array([0] + sorted(thresh[key]), dtype = float)
            ysm[key] = np.array(range(meds[key].shape[0]))/float(meds[key].shape[0]-1)
            yst[key] = np.array(range(thresh[key].shape[0]))/float(thresh[key].shape[0]-1)
        
        ## Plot Cumulative frequency graphs
        # Medians CF graph
        fig, ax1 = plt.subplots()
        # Get colours for different modalities
        modCols = {"unimodal": "aqua", "bimodal": "orange", "unclear": "lime"}
        # Get the minimum and maximum bin values for histograms if later desired
        minmed, maxmed = min([np.min(meds[m]) for m in meds.keys()]), max([np.max(meds[m]) for m in meds.keys()])
        minmed = math.floor(minmed/hist_width) * hist_width
        maxmed = math.ceil(maxmed/hist_width) * hist_width
        bins = np.arange(minmed, maxmed+hist_width, hist_width)

        # Plot graphs
        if(overlay_histogram):
            ax2 = ax1.twinx()
        for key in meds:
            ax1.plot(meds[key], ysm[key], color = modCols[key], label = f"{key.capitalize()} modality ({len(meds[key])-1})")
            # If a histogram overlay was desired, plot it
            if(overlay_histogram):
                h, edges = np.histogram(meds[key], bins)
                ax2.stairs(h, edges, color = modCols[key])
        ax1.set_xlabel("Median survivability correlation")
        ax1.set_ylabel("Cumulative frequency")
        if(overlay_histogram):
            ax2.set_ylabel("Histogram frequency")
        ax1.set_title(f"{mode.capitalize()} survivability scores median values")
        ax1.legend()
        plt.savefig(os.path.join(save_dir, f"{mode.capitalize()} survivability correlation median values by modality CDF.png"))
        plt.clf()
        plt.close()

        # Threshold CF graph
        fig, ax1 = plt.subplots()
        # Again, get maximum and minimum bin values
        minmed, maxmed = min([np.min(thresh[key]) for key in thresh.keys()]), max([np.max(thresh[key]) for key in thresh.keys()])
        minmed = math.floor(minmed/hist_width) * hist_width
        maxmed = math.ceil(maxmed/hist_width) * hist_width
        bins = np.arange(minmed, maxmed+hist_width, hist_width)

        # Plot graphs
        if(overlay_histogram):
            ax2 = ax1.twinx()
        for key in thresh:
            ax1.plot(thresh[key], yst[key], color = modCols[key], label = f"{key.capitalize()} modality ({len(thresh[key])-1})")
            # If a histogram overlay was desired, plot it
            if(overlay_histogram):
                h, edges = np.histogram(thresh[key], bins)
                ax2.stairs(h, edges, color = modCols[key])
        ax1.set_xlabel("'strong' survivability correlation threshold")
        ax1.set_ylabel("Cumulative frequency")
        ax2.set_ylabel("Histogram frequency")
        ax1.set_title(f"{mode.capitalize()} survivability scores threshold values")
        ax1.legend()
        plt.savefig(os.path.join(save_dir, f"{mode.capitalize()} survivability correlation threshold values by modality CDF.png"))
        plt.clf()
        plt.close()
        return
    
    def plot_high_survivors(self, mode: str = "drug", save_dir: str = os.path.join("Data", "Results", "modality graphs")):
        mode = mode.lower().strip()
        if(mode=="both"):
            self.plot_high_survivors("drug", save_dir)
            self.plot_high_survivors("gene", save_dir)
            return
        
        # Get a dictionary containing all the drug/gene survivability values
        if(mode=="drug"):
            survivability_arrays = {drug: self.datasets["AllByAll"].loc[[drug]].values for drug in self.datasets["AllByAll"].index}
        elif(mode=="gene"):
            survivability_arrays = {gene: self.datasets["AllByAll"][gene].values for gene in self.datasets["AllByAll"].columns}

        # Create dictionaries for results of medians and strong-correlation thresholds for different modality types
        thresholds = {"unimodal": {}, "bimodal": {}, "unclear": {}}

        # Get relevant data
        if(mode=="drug"):
            data = self.datasets["drug modality summary"]
        else:
            data = self.datasets["gene modality summary"]
        
        # Insert threshold values into dictionaries for easier access
        for mtype in data:
            # dg stands for drug-gene, as depending on the function mode this variable could be iterating over drugs or genes
            for dg in data[mtype].keys():
                thresholds[mtype][dg] = get_survivability_threshold(dg, data[mtype], survivability_arrays[dg])
        
        # Get rid of any NaN values
        for mtype in data:
            for dg in data[mtype]:
                if(thresholds[mtype][dg]!=thresholds[mtype][dg]):
                    del thresholds[mtype][dg]
        
        # Get counts of values above thresholds
        counts = {"unimodal": [], "bimodal": [], "unclear": []}
        for mtype in thresholds:
            for dg in thresholds[mtype]:
                arr = survivability_arrays[dg]
                # Remove NaN values from array
                arr = arr[~np.isnan(arr)]
                # Get number of values in array above threshold
                counts[mtype].append(arr[arr>=thresholds[mtype][dg]].shape[0])
        

        ## Make histograms
        maxcount = max((max(counts["unimodal"] + [0])+1, max(counts["bimodal"] + [0])+1, max(counts["unclear"] + [0])+1, 0))
        # Round up maxcount to nearest product of 5 + 1 (+1 as np.arange doesn't add the last value)
        maxcount += (maxcount%5) + 1
        bins = np.arange(0, maxcount, step = 5)
        maxcount, modalitycolours = -np.inf, {"unimodal": "blue", "bimodal": "orange", "unclear": "green"}
        # Store the numpy histogram data and max value
        # (This for loop needs to be run pre-emptively to get the maximum value of all types to ensure we get the maximum value of the entire graph,
        #  but it's inefficient to re-run np.histogram so I store the results for later use)
        histData = {}
        for mtype in counts:
            h, edges = np.histogram(counts[mtype], bins)
            histData[mtype] = {"h": h, "edges": edges}
            if(h.max()>maxcount):
                maxcount = h.max()
        # Set figure size
        plt.figure(figsize = (19.2, 14.4))
        for typei, mtype in enumerate(counts):
            h, edges = histData[mtype]["h"], histData[mtype]["edges"]
            plt.stairs(h, edges, label = f"{mtype} ({len(counts[mtype])})", color = modalitycolours[mtype])
        # Add axis labels
        plt.xlabel(f"Number of targets for a given {mode}")
        plt.ylabel("Frequency")
        plt.ylim((0, maxcount*1.05))
        plt.title(f"Histogram of {mode} strong target counts")
        plt.legend(loc = "center right")
        plt.savefig(os.path.join(save_dir, f"{mode} strong target count histogram.png"))
        return
    
    # Get modality data from the 'stats.json' file in results plots
    def __get_mod_data(self, mode: str) -> Optional[dict]:
        # Get relevant data
        if(mode=="drug"):
            return self.datasets["drug modality summary"]
        elif(mode=="gene"):
            return self.datasets["gene modality summary"]
        else:
            print(f"Unrecognised mode type: {mode}. Please use 'drug' or 'gene' as the mode.")
            return
    
    # Get arrays of survivability correlation values for targets of a given drug or gene
    def __get_survivability_arrays(self, mode: str) -> Optional[dict]:
        # Get a dictionary containing all the drug/gene survivability values
        if(mode=="drug"):
            return {drug: self.datasets["AllByAll"].loc[[drug]].values for drug in self.datasets["AllByAll"].index}
        elif(mode=="gene"):
            return {gene: self.datasets["AllByAll"][gene].values for gene in self.datasets["AllByAll"].columns}
        else:
            print(f"Unrecognised mode type: {mode}. Please use 'drug' or 'gene' as the mode.")
            return
    
    # Get nested dictionaries with all survivability correlation values
    def __get_survivability_dicts(self, mode: str) -> Optional[dict]:
        df = self.datasets["AllByAll"]
        if(mode=="drug"):
            return {drug: {gene: df.loc[[drug]][gene].values[0] for gene in df.columns} for drug in df.index}
        elif(mode=="gene"):
            return {gene: {drug: df.loc[[drug]][gene].values[0] for drug in df.index} for gene in df.columns}
        else:
            print(f"Unrecognised mode type: {mode}. Please use 'drug' or 'gene' as the mode.")
            return
    
    # Get counts for targets above threshold values
    def __get_counts(self, data: dict, survivability_arrays: dict) -> dict:
        # Make dictionary for threshold values
        thresh = {mtype: {} for mtype in data.keys()}
        # Get threshold values and insert them into dictionaries
        for mtype in data:
            # dg stands for drug-gene, as depending on the function mode this variable could be iterating over drugs or genes
            for dg in data[mtype].keys():
                thresh[mtype][dg] = get_survivability_threshold(dg, data[mtype], survivability_arrays[dg])
        
        # Get rid of any NaN values
        toremove = {mtype: [] for mtype in thresh.keys()}
        for mtype in thresh:
            for dg in thresh[mtype]:
                if(thresh[mtype][dg]!=thresh[mtype][dg]):
                    toremove[mtype].append(dg)
        for mtype in toremove:
            for tr in toremove[mtype]:
                del thresh[mtype][tr]
        
        # Get number of targets above threshold values
        counts = {mtype: {} for mtype in thresh.keys()}
        for mtype in thresh:
            for dg in thresh[mtype]:
                arr = survivability_arrays[dg]
                counts[mtype][dg] = arr[arr>thresh[mtype][dg]].shape[0]
        
        return counts

    
    # Plot waterfall graphs for each modality type, plotting number of targets for each drug
    def plot_waterfall(self, mode: str = "drug", save_dir = os.path.join("Data", "Results", "modality graphs"),
                       keep_unclear: bool = False):
        mode = mode.lower().strip()
        if(mode=="both"):
            self.plot_waterfall("drug", save_dir, keep_unclear)
            self.plot_waterfall("gene", save_dir, keep_unclear)
            return
        
        # Get a dictionary containing all the drug/gene survivability values
        survivability_arrays = self.__get_survivability_arrays(mode)

        # Get relevant data
        data = self.__get_mod_data(mode)
        
        # Remove 'unclear' as an option if desired
        if(not keep_unclear):
            del data["unclear"]

        counts = self.__get_counts(data, survivability_arrays)
        
        # Sort into sorted arrays tuples ({modalityType: [(drug, count(lowest)), (drug2, count2), ..., (drugn, countn(highest))]})
        for mtype in counts:
            stlist = [(key, val) for val, key in sorted(zip(list(counts[mtype].values()), list(counts[mtype].keys())))]
            counts[mtype] = deepcopy(stlist)
        
        # Plot waterfall plots
        mirror = {"drug": "gene", "gene": "drug"}[mode]
        for mtype in counts:
            stlist = counts[mtype][::-1]
            plt.figure(figsize=(19.2, 14.4))
            plt.bar([stlist[i][0] for i in range(len(stlist))], [stlist[i][1] for i in range(len(stlist))])
            plt.title(f"{mode.capitalize()}-{mirror} strong targets")
            plt.xlabel(mode.capitalize())
            plt.ylabel(f"Strong {mirror} target count")
            # Remove xticks
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            plt.savefig(os.path.join(save_dir, f"{mode} target waterfall plot {mtype}.png"))
            plt.clf()
            plt.close()
        return
    
    def plot_compare_targets(self, mode: str = "drug", save_dir = os.path.join("Data", "Results", "modality graphs"),
                       keep_unclear: bool = False) -> None:
        return


def get_survivability_threshold(dg: str, SurvivabilityDict: dict, survivability_array: Optional[np.ndarray] = None) -> float:
    # First, clear any NaN values out of survivability array (assuming it's available)
    if(survivability_array is not None):
        survivability_array = survivability_array[~np.isnan(survivability_array)]
    # Try to use the available data to get the appropriate cutoff
    try:
        rel = SurvivabilityDict[dg]
        # If the modality is unimodal or unclear, use 3 SDs above the mean as the threshold
        if(rel["modality details"]["modality"]!="bimodal"):
            thresh = float(rel["modality details"]["mean"]) + float(rel["standard deviations"]["3.0"])
        # If the modality is bimodal, use the mean of the higher curve survivability as the threshold
        else:
            if(float(rel["curve parameters"]["mu0"])>float(rel["curve parameters"]["mu1"])):
                thresh = float(rel["curve parameters"]["mu0"])+(3*float(rel["curve parameters"]["sigma0"]))
            else:
                thresh = float(rel["curve parameters"]["mu1"])+(3*float(rel["curve parameters"]["sigma1"]))
        # If the array of survivability scores is available, make some tweaks to improve appropriate threshold determination
        if(survivability_array is not None):
            # make sure there's at least SOMETHING above the threshold score, otherwise correct to 3SDs above mean
            # alternatively, correct to 3SDs if more than 10% of correlations are strong
            if(thresh>np.max(survivability_array) or (survivability_array[survivability_array>thresh]).shape[0]>survivability_array.shape[0]*0.1):
                thresh = np.mean(survivability_array) + (3*np.std(survivability_array))
            # If the threshold is still too high, correct to the maximum score (i.e. gives a single target equal to the threshold score)
            thresh = min(thresh, np.max(survivability_array))
            # If the threshold means more tha 5% of the targets cross the threshold, curb this to 5%
            if((survivability_array[survivability_array>thresh]).shape[0]>survivability_array.shape[0]*0.05):
                thresh = np.quantile(survivability_array, 0.95)
        # Return the threshold rounded down to 3 dp
        return round_down(thresh, 3)
    # If unsuccessful, try to just use 3 SDs above the mean
    except Exception as e:
        print(f"Failed to retrieve modality information for {dg}; defaulting to 3SDs above norm...")
        return float(np.mean(survivability_array)) + (float(np.std(survivability_array))*3.)

def round_down(f, dp: int) -> float:
    rounded = round(f, dp)
    if(rounded>f):
        rounded -= np.power(10., -1*dp)
    return rounded
