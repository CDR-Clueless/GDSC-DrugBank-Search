#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2025

@author: jds40
"""

import os
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Union, Optional
from copy import deepcopy
from tqdm import tqdm
import matplotlib as mpl

from data_handler import DataHandler
from drugbank_handler import DrugbankHandler
from drug_search import update_hgnc_single
from modality_analysis import get_survivability_threshold
COMBINATION_DATA_DIR: str = os.path.join("Data", "Raw Data", "GDSCC")

class CombinationAnalyser(DataHandler):
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
    
    def __import_raw_data(self, folderDir: str = COMBINATION_DATA_DIR):
        # Establish main DataFrame to be added to
        df = pd.DataFrame(data = None, columns = ["Combo Name", "Cell Line Name", "Left Drug eMax", "Right Drug eMax", "Combo eMax"])
        # Keep track of files which have been loaded
        loaded_files = []
        # Go through available files in the directory
        for filename in os.listdir(folderDir):
            # Check if this file is a GDSCC data file
            if(("anchor" in filename.lower() or "matrix" in filename.lower()) and filename.lower()[-4:] in [".csv", ".tsv"]):
                # Get appropriate separator for loading the DataFrame
                sep = {"csv": ",", "tsv": "\t"}[filename.lower()[-3:]]
                # Load the data into a consistent, more standardised format
                newdf = pd.read_csv(os.path.join(folderDir, filename), sep = sep, low_memory=False)
                if("anchor" in filename.lower()):
                    # Refine to relevant columns
                    newdf = newdf[["Anchor Name", "Library Name", "Cell Line name", "Bliss Emax", "Library Emax", "Combo Emax"]]
                    # Capitalise drug names to improve how standard they are
                    newdf["Anchor Name"] = newdf["Anchor Name"].apply(lambda x: x.upper().strip())
                    newdf["Library Name"] = newdf["Library Name"].apply(lambda x: x.upper().strip())
                    # Combine Anchor and Library names into Combo name which is standardised by alphabet so duplicates can be removed later
                    rows = []
                    for i in range(len(newdf)):
                        anchor, library = newdf["Anchor Name"].values[i], newdf["Library Name"].values[i]
                        if(anchor>library):
                            rows.append(deepcopy([f"{anchor}###{library}", newdf["Cell Line name"].values[i],
                                                newdf["Bliss Emax"].values[i], newdf["Library Emax"].values[i],
                                                newdf["Combo Emax"].values[i]]))
                        else:
                            rows.append(deepcopy([f"{library}###{anchor}", newdf["Cell Line name"].values[i],
                                                newdf["Library Emax"].values[i], newdf["Bliss Emax"].values[i],
                                                newdf["Combo Emax"].values[i]]))
                    # Add this data into the main DataFrame
                    df = pd.concat([df, pd.DataFrame(data = rows, columns = ["Combo Name", "Cell Line Name",
                                                                            "Left Drug eMax", "Right Drug eMax",
                                                                            "Combo eMax"])], ignore_index=True)
                    loaded_files.append(filename)
                elif("matrix" in filename.lower()):
                    # Refine to relevant columns
                    newdf = newdf[["lib1_name", "lib2_name", "CELL_LINE_NAME", "lib1_MaxE", "lib2_MaxE", "combo_MaxE"]]
                    # Capitalise drug names to improve standardisation
                    newdf["lib1_name"] = newdf["lib1_name"].apply(lambda x: x.upper().strip())
                    newdf["lib2_name"] = newdf["lib2_name"].apply(lambda x: x.upper().strip())
                    # Combine library names into combo name which is standardised by alphabet so duplicates can be removed later
                    rows = []
                    for i in range(len(newdf)):
                        l1, l2, cln, l1emax, l2emax, cemax = [newdf[c].values[i] for c in 
                                                            ["lib1_name", "lib2_name", "CELL_LINE_NAME",
                                                            "lib1_MaxE", "lib2_MaxE", "combo_MaxE"]]
                        if(l1>l2):
                            rows.append(deepcopy([f"{l1}###{l2}", cln, l1emax, l2emax, cemax]))
                        else:
                            rows.append(deepcopy([f"{l2}###{l1}", cln, l2emax, l1emax, cemax]))
                    # Add this to the main DataFrame
                    df = pd.concat([df, pd.DataFrame(data = rows, columns = ["Combo Name", "Cell Line Name",
                                                                            "Left Drug eMax", "Right Drug eMax",
                                                                            "Combo eMax"])], ignore_index=True)
                    loaded_files.append(filename)
        self.datasets["raw data"] = deepcopy(df)
    
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
    
    # Get Bliss Indepndence scores of each drug combination
    def bliss_independence(self):
        # Make sure raw data is available
        if("raw data" not in self.datasets.keys()):
            self.__import_raw_data()
        df = self.datasets["raw data"]
        df["Bliss Independence"] = np.divide(df["Left Drug eMax"]+df["Right Drug eMax"]-(df["Left Drug eMax"]*df["Right Drug eMax"]), df["Combo eMax"])
        return df
    
    # Compare the medians and threshold SC values to bliss independence scores
    def bliss_sc_comparison(self, mode = "both", saveDir = os.path.join("Data", "Results", "combination analysis")):
        # Strip mode
        mode = mode.replace(" ","").lower()
        # Check mode
        if(mode=="both"):
            self.bliss_sc_comparison(mode = "threshold")
            self.bliss_sc_comparison(mode = "median")
            return
        elif(mode in ["threshold", "median"]):
            # Get Bliss independence scores
            bliss = self.bliss_independence()
            # Get DataFrame with information on SC thresholds
            results = {}
            for key in ["Combo", "Left Drug", "Right Drug"]:
                results[key] = {}
                data = self.datasets[key]
                for col in data:
                    # Clear out NaN values
                    survivability_array = np.array(data[col].values, dtype = float)
                    survivability_array = survivability_array[~np.isnan(survivability_array)]
                    if(mode=="threshold"):
                        results[key][col] = get_survivability_threshold(col, self.datasets["drug modality summary"], survivability_array)
                    else:
                        results[key][col] = np.quantile(survivability_array, 0.5)
            if(mode=="median"):
                print(results)
            df = pd.DataFrame(data = results)
            df["CL"] = df["Combo"] > df["Left Drug"]
            df["RL"] = df["Combo"] > df["Right Drug"]
            df["Combo Comparison"] = df["CL"].astype(int)+df["RL"].astype(int)
            del df["CL"]; del df["RL"]
            cm = {key: {"Lower": 0, "Between": 0, "Higher": 0} for key in ["Above 1", "Below 1"]}
            for dc in df.index:
                rel = bliss.loc[bliss["Combo Name"]==dc]
                higher = len(rel.loc[rel["Bliss Independence"]>1])
                lower = len(rel.loc[rel["Bliss Independence"]<1])
                val = df.loc[df.index==dc]["Combo Comparison"].values[0]
                if(val==0):
                    cm["Above 1"]["Lower"] += higher
                    cm["Below 1"]["Lower"] += lower
                elif(val==1):
                    cm["Above 1"]["Between"] += higher
                    cm["Below 1"]["Between"] += lower
                elif(val==2):
                    cm["Above 1"]["Higher"] += higher
                    cm["Below 1"]["Higher"] += lower
            # Plot Confusion Matrix
            fig, ax = plt.subplots()
            plt.grid(False)
            sns.set_theme()
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "sans-serif",
                'pgf.preamble': r'\\usepackage{amsmath}'
            })
            formatted = np.array([list(cm[key].values()) for key in cm.keys()], dtype = int).T
            im = ax.imshow(formatted)
            fig.colorbar(im)
            # Add LaTeX labels
            ax.set_yticks([0, 1, 2])
            baseyticks = [r"$C^{SC_{REPLACEHERE}} < D_{1}^{SC_{REPLACEHERE}} \wedge D_{2}^{SC_{REPLACEHERE}}$",
                            r"$C^{SC_{REPLACEHERE}} < D_{1}^{SC_{REPLACEHERE}} \vee D_{2}^{SC_{REPLACEHERE}}$",
                            r"$C^{SC_{REPLACEHERE}} > D_{1}^{SC_{REPLACEHERE}} \wedge D_{2}^{SC_{REPLACEHERE}}$"]
            if(mode=="threshold"):
                ax.set_yticklabels([label.replace("REPLACEHERE", "T") for label in baseyticks])
            else:
                ax.set_yticklabels([label.replace("REPLACEHERE", r"\mu 1/2") for label in baseyticks])
            ax.set_xticks([0, 1])
            ax.set_xticklabels([r"$B > 1$", r"$B < 1$"])
            # Add title depending on current mode
            if(mode=="threshold"):
                ax.set_title("Survivability Correlation Threshold Comparison")
            else:
                ax.set_title("Survivability Correlation Median Comparison")
            # Add numbers to the squares
            for i in range(formatted.shape[0]):
                for j in range(formatted.shape[1]):
                    ax.text(j, i, str(formatted[i][j]), color = "blue", verticalalignment = "center", horizontalalignment = "center")
            # Add X any Y labels
            plt.xlabel("Bliss Independence Score")
            if(mode=="threshold"):
                plt.ylabel("Survivability Correlation Threshold Value Comparison")
            else:
                plt.ylabel("Survivability Correlation Median Value Comparison")
            # Save figure
            if(mode=="threshold"):
                plt.savefig(os.path.join(saveDir, "SC Threshold-Bliss Independence Confusion Matrix.png"))
            else:
                plt.savefig(os.path.join(saveDir, "SC Median-Bliss Independence Confusion Matrix.png"))
            plt.clf()
            plt.close()
        return