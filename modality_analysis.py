#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 11 Dec 2025

@author: jds40
"""

import os
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Union, Optional
from copy import deepcopy
from tqdm import tqdm

from data_handler import DataHandler
from drugbank_handler import DrugbankHandler
from drug_search import update_hgnc_single
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
        # First, ensure the directory is valid
        if(not os.path.exists(save_dir)):
            os.mkdir(save_dir)
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
        if(not keep_unclear and "unclear" in data.keys()):
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
            # Note: the minimum is repeated at the start as this point forms the start of the graph (otherwise the minimimum x value would have a y of 0)
            # This used to be adding a 0 so the graph started at origin but this caused issues with x-axes starting below zero
            meds[key] = np.array([np.min(meds[key])] + sorted(meds[key]), dtype = float)
            thresh[key] = np.array([np.min(thresh[key])] + sorted(thresh[key]), dtype = float)
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
        if(not keep_unclear and "unclear" in data.keys()):
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
        mode = mode.lower().strip()
        if(mode=="both"):
            self.plot_compare_targets("drug", save_dir, keep_unclear)
            self.plot_compare_targets("gene", save_dir, keep_unclear)
            return
        elif(mode not in ["drug", "gene", "both"]):
            print(f"Unrecognised mode: {mode}. Please use 'drug', 'gene', or 'both' as the mode argument.")
            return
        
        # Get a dictionary containing all the drug/gene survivability values
        dbg = self.datasets["AllByAll"]

        # Get relevant data
        data = self.__get_mod_data(mode)
        
        # Remove 'unclear' as an option if desired
        if(not keep_unclear and "unclear" in data.keys()):
            del data["unclear"]

        # Initialize a drug bank handler class to find the relevant drugbank targets
        dbh = DrugbankHandler()

        hgnc = pd.read_table(os.path.join("Data", "Laurence-Data", "hgnc_complete_set.tsv"), low_memory=False).fillna('')

        # Go through drugs/genes
        missing, results = [], {mtype: {} for mtype in data.keys()}
        for mtype in data:
            for dg in tqdm(data[mtype], desc = f"Processing drugs of {mtype} modality"):
                # Get actual, primary targets from DrugBank
                realTargets = dbh.fetch_targets(dg, mode)
                # If no real targets are found, continue the loop
                if(len(realTargets)<1):
                    missing.append(dg)
                    continue
                # Get all drugs/genes and their scores
                if(mode=="drug"):
                    corrs = dbg.loc[[dg]]
                    corrDict = {gene: corrs[gene].values[0] for gene in corrs.columns}
                else:
                    corrs = dbg[dg]
                    corrDict = {drug: corrs.loc[[drug]].values[0] for drug in corrs.index}
                # Get the threshold for strong targets
                thresh = get_survivability_threshold(dg, data[mtype], np.array(list(corrDict.values()), dtype = float))
                # Find strong targets
                targets = [target for target in corrDict.keys() if corrDict[target]>=thresh]
                # standardise the strong target names if necessary
                if(mode=="drug"):
                    for i in range(len(targets)):
                        targets[i] = update_hgnc_single(targets[i], hgnc)
                sim = 0
                for target in realTargets:
                    if(target in targets):
                        sim += 1
                results[mtype][dg] = sim / len(realTargets)
        
        with open(os.path.join(os.path.dirname(save_dir), "drugbankDrugs.txt"), "w") as f:
            f.write("\n".join(dbh.fetch_drugs()))
        with open(os.path.join(os.path.dirname(save_dir), "correlationDrugs.txt"), "w") as f:
            f.write("\n".join(list(dbg.index)))

        return

class SquaredModalityAnalyzer(DataHandler):
    def __init__(self, datasets: Union[list, tuple, None] = None):
        super().__init__(datasets)
        if("drug modality summary" not in self.datasets.keys()):
            self.load_data("drug modality summary", os.path.join("Data", "Results", "GDSCC drug-gene correlation frequency histograms", "stats.json"))
        #if("gene modality summary" not in self.datasets.keys()):
            #self.load_data("gene modality summary", os.path.join("Data", "Results", "Gene-drug correlation frequency histograms", "stats.json"))
        for side in ["Combo", "Left Drug", "Right Drug"]:
            if(side not in self.datasets.keys()):
                self.load_data(side, os.path.join("Data", "Results", "Survivability-Correlations", f"GDSCC-{side} eMax-AllDrugsByAllGenes.tsv"))
                self.datasets[side].set_index("symbol", inplace = True)
        # Sort internal datasets into unique modalities
        self.__sort_modalities()
        return

    # Sort internal dataset into unimodal, unclear and bimodal dictionaries
    def __sort_modalities(self):
        for internalname in ["drug modality summary", "gene modality summary"]:
            if(internalname not in self.datasets.keys()):
                print(f"{internalname} not found in datasets; cannot sort modalities")
                continue
            uni, unc, bim = [deepcopy({"Combo": {}, "Left": {}, "Right": {}}) for _ in range(3)]
            for drugCombo in self.datasets[internalname]:
                for side in self.datasets[internalname][drugCombo]:
                    deets = self.datasets[internalname][drugCombo][side]["modality details"]["modality"]
                    newdata = self.datasets[internalname][drugCombo][side]
                    if(deets=="unimodal"):
                        uni[side][drugCombo] = deepcopy(newdata)
                    elif(deets=="unclear"):
                        unc[side][drugCombo] = deepcopy(newdata)
                    elif(deets=="bimodal"):
                        bim[side][drugCombo] = deepcopy(newdata)
                    else:
                        print(f"Could not sort unrecognised modality: {deets}")
            # Add this sorted entry
            self.datasets[internalname + " organised"] = {"unimodal": deepcopy(uni), "bimodal": deepcopy(bim), "unclear": deepcopy(unc)}
        return

    # Get modality data from the 'stats.json' file in results plots
    def __get_mod_data(self, mode: str, organised: bool = False) -> Optional[dict]:
        # Get relevant data
        if(mode=="drug"):
            return self.datasets["drug modality summary" + {True:" organised", False:""}[organised]]
        elif(mode=="gene"):
            return self.datasets["gene modality summary" + {True:" organised", False:""}[organised]]
        else:
            print(f"Unrecognised mode type: {mode}. Please use 'drug' or 'gene' as the mode.")
            return
    
    # Get arrays of survivability correlation values for targets of a each drug/gene for left, right and combination of drugs
    def __get_survivability_arrays(self, mode: str) -> Optional[dict]:
        if(mode=="drug"):
            return {side.replace(" Drug",""): {drugs: self.datasets[side][drugs].values for drugs in self.datasets[side].columns} for side in ["Combo", "Left Drug", "Right Drug"]}
        elif(mode=="gene"):
            return {side.replace(" Drug",""): {gene: self.datasets[side].loc[[gene]].values for gene in self.datasets[side].index} for side in ["Combo", "Left Drug", "Right Drug"]}
        else:
            print(f"Unrecognised mode type: {mode}. Please use 'drug' or 'gene' as the mode.")
            return
    
    # Get counts for targets above threshold values
    def __get_counts(self, data: dict, survivability_arrays: dict) -> dict:
        # Make dictionary for threshold values
        thresh = {mtype: {side: {} for side in data[mtype].keys()} for mtype in data.keys()}
        # Get threshold values and insert them into dictionaries
        for mtype in data:
            for side in data[mtype]:
                # dg stands for drug-gene, as depending on the function mode this variable could be iterating over drugs or genes
                for dg in data[mtype][side].keys():
                    thresh[mtype][side][dg] = get_survivability_threshold(dg, data[mtype][side], survivability_arrays[side][dg])
        
        # Get rid of any NaN values
        toremove = {mtype: {side: [] for side in thresh[mtype].keys()} for mtype in thresh.keys()}
        for mtype in thresh:
            for side in thresh[mtype]:
                for dg in thresh[mtype][side]:
                    if(thresh[mtype][side][dg]!=thresh[mtype][side][dg]):
                        toremove[mtype][side].append(dg)
        for mtype in toremove:
            for side in toremove[mtype]:
                for tr in toremove[mtype][side]:
                    del thresh[mtype][side][tr]
        
        # Get number of targets above threshold values
        counts = {mtype: {side: {} for side in thresh[mtype].keys()} for mtype in thresh.keys()}
        for mtype in thresh:
            for side in thresh[mtype]:
                for dg in thresh[mtype][side]:
                    arr = survivability_arrays[side][dg]
                    counts[mtype][side][dg] = arr[arr>thresh[mtype][side][dg]].shape[0]
        
        return counts
    
    # Plot cumulative frequency graphs for each modality type, plotting medians and threshold values
    def plot_cf(self, mode: str = "drug", save_dir = os.path.join("Data", "Results", "squared modality graphs"),
                keep_unclear: bool = False, overlay_histogram: bool = True, hist_width: float = 0.025):
        # First, ensure the directory is valid
        if(not os.path.exists(save_dir)):
            os.mkdir(save_dir)
        mode = mode.lower().strip()
        if(mode=="both"):
            self.plot_cf("drug", save_dir, keep_unclear, overlay_histogram)
            self.plot_cf("gene", save_dir, keep_unclear, overlay_histogram)
            return
        
        # Get a dictionary containing all the drug/gene survivability values
        survivability_arrays = self.__get_survivability_arrays(mode)
        if(survivability_arrays is None):
            return

        # Create dictionaries for results of medians and strong-correlation thresholds for different modality types
        meds, thresh = {}, {}
        for side in ["Combo", "Left", "Right"]:
            meds[side], thresh[side] = {}, {}
            for mod in ["unimodal", "bimodal", "unclear"]:
                meds[side][mod], thresh[side][mod] = [], []

        # Get relevant data
        if(mode=="drug"):
            data = self.datasets["drug modality summary organised"]
        else:
            data = self.datasets["gene modality summary organised"]
        
        # Remove 'unclear' as an option if desired
        if(not keep_unclear and "unclear" in data.keys()):
            for side in meds:
                del meds[side]["unclear"]
                del thresh[side]["unclear"]
            del data["unclear"]

        # Insert relevant data into dictionaries - the median and threshold values
        for mtype in data:
            for side in data[mtype]:
                # dg stands for drug-gene, as depending on the function mode this variable could be iterating over drugs or genes
                for dg in data[mtype][side].keys():
                    meds[side][mtype].append(np.quantile(survivability_arrays[side][dg][~np.isnan(survivability_arrays[side][dg])], 0.5))# float(data[mtype][dg]["quantiles"]["0.5"]))
                    thresh[side][mtype].append(get_survivability_threshold(dg, data[mtype][side], survivability_arrays[side][dg]))
        
        # Combine the individual drugs into single arrays, convert the med and thresh lists into arrays, Get rid of any NaN values and sort them
        meds["Single Drug"], thresh["Single Drug"] = {}, {}
        for mtype in data:
            meds["Combo"][mtype] = np.array(sorted(meds["Combo"][mtype]), dtype = float)
            meds["Combo"][mtype] = meds["Combo"][mtype][~np.isnan(meds["Combo"][mtype])]
            thresh["Combo"][mtype] = np.array(sorted(thresh["Combo"][mtype]), dtype = float)
            thresh["Combo"][mtype] = thresh["Combo"][mtype][~np.isnan(thresh["Combo"][mtype])]
            meds["Single Drug"][mtype] = np.array(sorted(meds["Left"][mtype] + meds["Right"][mtype]), dtype = float)
            meds["Single Drug"][mtype] = meds["Single Drug"][mtype][~np.isnan(meds["Single Drug"][mtype])]
            thresh["Single Drug"][mtype] = np.array(sorted(thresh["Left"][mtype] + thresh["Right"][mtype]), dtype = float)
            thresh["Single Drug"][mtype] = thresh["Single Drug"][mtype][~np.isnan(thresh["Single Drug"][mtype])]
        del meds["Left"]
        del meds["Right"]
        del thresh["Left"]
        del thresh["Right"]
        
        # Format the values and into x and y cumulative frequency values
        ysm, yst = [deepcopy({"Combo": {}, "Single Drug": {}}) for _ in range(2)]
        for side in meds:
            for mod in meds[side]:
                # Note: the minimum is repeated at the start as this point forms the start of the graph (otherwise the minimimum x value would have a y of 0)
                # This used to be adding a 0 so the graph started at origin but this caused issues with x-axes starting away from 0, especially starting at a negative
                meds[side][mod] = np.concat(([np.min(meds[side][mod])], meds[side][mod]))
                thresh[side][mod] = np.concat(([np.min(thresh[side][mod])], thresh[side][mod]))
                ysm[side][mod] = np.array(range(meds[side][mod].shape[0]))/float(meds[side][mod].shape[0]-1)
                yst[side][mod] = np.array(range(thresh[side][mod].shape[0]))/float(thresh[side][mod].shape[0]-1)
        
        ## Plot Cumulative frequency graphs
        for mod in meds["Combo"]:
            # Medians CF graph
            fig, ax1 = plt.subplots()
            # Get colours for different modalities
            modCols = {"Single Drug": "aqua", "Combo": "lime"}
            # Get the minimum and maximum bin values for histograms and create the bins from this
            minmed, maxmed = np.inf, -np.inf
            for side in meds:
                minmed = min(minmed, np.min(meds[side][mod]))
                maxmed = max(maxmed, np.max(meds[side][mod]))
            minmed = math.floor(minmed/hist_width) * hist_width
            maxmed = math.ceil(maxmed/hist_width) * hist_width
            bins = np.arange(minmed, maxmed+hist_width, hist_width)

            # Plot graphs
            if(overlay_histogram):
                ax2 = ax1.twinx()

            for side in meds:
                ax1.plot(meds[side][mod], ysm[side][mod], color = modCols[side], label = f"{side.replace('Combo','Both Drugs').capitalize()} used ({len(meds[side][mod])-1})")
                # If a histogram overlay was desired, plot it
                if(overlay_histogram):
                    h, edges = np.histogram(meds[side][mod], bins)
                    ax2.stairs(h, edges, color = modCols[side])
            ax1.set_xlabel("Median survivability correlation")
            ax1.set_ylabel("Cumulative frequency")
            if(overlay_histogram):
                ax2.set_ylabel("Histogram frequency")
            ax1.set_title(f"{mod.capitalize()} survivability scores median values")
            ax1.legend()
            plt.savefig(os.path.join(save_dir, f"{mod.capitalize()} survivability correlation median values by modality CDF.png"))
            plt.clf()
            plt.close()

            # Threshold CF graph
            fig, ax1 = plt.subplots()
            # Again, get maximum and minimum bin values
            minmed, maxmed = np.inf, -np.inf
            for side in thresh:
                minmed = min(minmed, np.min(thresh[side][mod]))
                maxmed = max(maxmed, np.max(thresh[side][mod]))
            minmed = math.floor(minmed/hist_width) * hist_width
            maxmed = math.ceil(maxmed/hist_width) * hist_width
            bins = np.arange(minmed, maxmed+hist_width, hist_width)

            # Plot graphs
            if(overlay_histogram):
                ax2 = ax1.twinx()
            for side in thresh:
                ax1.plot(thresh[side][mod], yst[side][mod], color = modCols[side], label = f"{side.replace('Combo','Both Drugs').capitalize()} used ({len(thresh[side][mod])-1})")
                # If a histogram overlay was desired, plot it
                if(overlay_histogram):
                    h, edges = np.histogram(thresh[side][mod], bins)
                    ax2.stairs(h, edges, color = modCols[side])
            ax1.set_xlabel("'strong' survivability correlation threshold")
            ax1.set_ylabel("Cumulative frequency")
            ax2.set_ylabel("Histogram frequency")
            ax1.set_title(f"{mod.capitalize()} survivability scores threshold values")
            ax1.legend()
            plt.savefig(os.path.join(save_dir, f"{mod.capitalize()} survivability correlation threshold values by modality CDF.png"))
            plt.clf()
            plt.close()
        return
    
    # Plot waterfall graphs for each modality type, plotting number of targets for each drug
    def plot_waterfall(self, mode: str = "drug", save_dir = os.path.join("Data", "Results", "squared modality graphs"),
                       keep_unclear: bool = False):
        mode = mode.lower().strip()
        if(mode=="both"):
            self.plot_waterfall("drug", save_dir, keep_unclear)
            self.plot_waterfall("gene", save_dir, keep_unclear)
            return
        
        # Get a dictionary containing all the drug/gene survivability values
        survivability_arrays = self.__get_survivability_arrays(mode)
        if(survivability_arrays is None):
            return

        # Get relevant data
        data = self.__get_mod_data(mode, organised = True)
        
        # Remove 'unclear' as an option if desired
        if(not keep_unclear and "unclear" in data.keys()):
            del data["unclear"]

        counts = self.__get_counts(data, survivability_arrays)
        
        # Sort into sorted arrays tuples ({modalityType: [(drug, count(lowest)), (drug2, count2), ..., (drugn, countn(highest))]})
        for mtype in counts:
            # Organise left and right drugs into single drugs
            tr = {"Right-"+drugs: counts[mtype]["Right"][drugs] for drugs in counts[mtype]["Right"].keys()}
            tl = {"Left-"+drugs: counts[mtype]["Left"][drugs] for drugs in counts[mtype]["Left"].keys()}
            counts[mtype]["Single drug"] = deepcopy(tr | tl)
            counts[mtype]["Both drugs"] = deepcopy(counts[mtype]["Combo"])
            del counts[mtype]["Combo"]
            del counts[mtype]["Left"]
            del counts[mtype]["Right"]
            # Get sorted lists of these drugs
            for dc in ["Single drug", "Both drugs"]:
                stlist = [(key, val) for val, key in sorted(zip(list(counts[mtype][dc].values()), list(counts[mtype][dc].keys())))]
                counts[mtype][dc] = deepcopy(stlist)
        
        # Plot waterfall plots
        mirror = {"drug": "gene", "gene": "drug"}[mode]
        colours = {"Single drug": "aqua", "Both drugs": "green"}
        for mtype in counts:
            fig, ax = plt.subplots(figsize=(19.2, 14.4))
            for dc in ["Single drug", "Both drugs"]:
                # Get high-low ordered list of counts
                stlist = counts[mtype][dc][::-1]
                # Get 0-1 x-values and the appropriate bar widths for this graph
                xs = np.array(range(len(stlist)), dtype = float) / len(stlist)
                width = 1/(len(stlist))
                xs += (width/2)
                ax.bar(xs, [stlist[i][1] for i in range(len(stlist))], label = dc, width = width, edgecolor = colours[dc], fill = False)
            plt.title(f"{mtype} {mode.capitalize()}-{mirror} strong targets")
            plt.xlabel(mode.capitalize())
            plt.ylabel(f"Strong {mirror} target count")
            plt.legend()
            # Remove xticks
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            plt.savefig(os.path.join(save_dir, f"{mtype} {mode} target waterfall plot.png"))
            plt.clf()
            plt.close()
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
