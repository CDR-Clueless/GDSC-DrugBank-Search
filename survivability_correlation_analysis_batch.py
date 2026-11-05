#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 11 Aug 2026

@author: jds40
"""

import os
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from target_functions import get_drugTargets

DEFAULT_FILE_LOC: str = os.path.join("Data", "Results", "Survivability-Correlations", "GDSC")

PLOT_Z3: bool = False

def main():
    toImport = get_sc_files()
    # Remove Undesired files
    for key in list(toImport.keys()):
        if("_" in key):
            if(int(key.split("_")[1]) > 1):
                del toImport[key]
    # Get Targets
    dTargets = prepare_targetFrame()
    dTargets["TARGET"] = dTargets["TARGET"].replace("", np.nan)
    #print(len(dTargets["DRUG_STANDARD"].unique()))
    dTargets.dropna(axis = 0, subset = "TARGET", inplace = True)
    #print(len(dTargets["DRUG_STANDARD"].unique()))
    # Drop targets not in SC data
    df = pd.read_csv(toImport[list(toImport.keys())[0]], sep = "\t")
    df.set_index("symbol", inplace = True)
    drugs = [col.upper().replace(" ","").replace(" ","").replace("_", "").replace("(","").replace(")","") for col in df.columns]
    dTargets = dTargets.loc[dTargets["DRUG_STANDARD"].isin(drugs)]
    #print(len(dTargets["DRUG_STANDARD"].unique()))
    #print(list(df.index))
    #print(dTargets["TARGET"])
    dTargets = dTargets.loc[dTargets["TARGET"].isin(df.index)]
    dTargets.reset_index(inplace=True)
    #print(len(dTargets["DRUG_STANDARD"].unique()))
    # Add Z-Scores from each toImport DataFrame to dTargets
    for key in toImport:
        scData = pd.read_csv(toImport[key], sep = "\t")
        scData.set_index("symbol", inplace = True)
        scData.columns = [col.upper().replace(" ","").replace(" ","").replace("_", "").replace("(","").replace(")","") for col in scData.columns]
        dTargets = add_zscore(dTargets, scData, f"Z-{key}")
    # Get array of Z-Scores for known targets
    # Get number of scores above Z >= 1.645 and Z >= 3.0
    ZScores: dict = {}
    for col in dTargets.columns:
        if("Z-" not in col):
            continue
        name = col.replace("Z-","")
        # Remove non-float values
        rel = dTargets[col].dropna()
        vals = []
        for val in rel.values:
            try:
                vals.append(float(val))
            except:
                pass
        arr = np.array(vals, dtype = float)
        ZScores[name] = deepcopy(arr)

    if(PLOT_Z3):
        p5ticks = np.array(range(len(ZScores)), dtype = float) - 0.2
        p99ticks = np.array(range(len(ZScores)), dtype = float) + 0.2
        plt.bar(p5ticks, [sum(arr>=1.645) for arr in ZScores.values()], width = 0.4, color = "blue", label = "p < 0.05")
        plt.bar(p99ticks, [sum(arr>=3.0) for arr in ZScores.values()], width = 0.4, color = "orange", label = "Z > 3.0")
        plt.xticks(range(len(ZScores)), list(ZScores.keys()))
        plt.ylim((0, max([sum(arr>=1.645) for arr in ZScores.values()])+5))
        plt.legend()
    else:
        p5ticks = np.array(range(len(ZScores)), dtype = float)
        plt.bar(p5ticks, [sum(arr>=1.645) for arr in ZScores.values()], width = 0.8, color = "blue", label = "p < 0.05")
        plt.xticks(range(len(ZScores)), list(ZScores.keys()))
        plt.ylim((0, max([sum(arr>=1.645) for arr in ZScores.values()])+5))
    plt.show()

    plt.boxplot(list(ZScores.values()), tick_labels=list(ZScores.keys()))
    plt.show()
    return

def get_sc_files(fileDir: str = DEFAULT_FILE_LOC):
    if(not os.path.exists(fileDir)):
        print(f"Directory not found: {fileDir}")
        return {}
    toReturn: dict = {}
    for filename in os.listdir(fileDir):
        if("alldrugsbyallgenes.tsv" in filename.lower()):
            x = filename.split("-")
            toReturn[f"{x[1]}"] = os.path.join(fileDir, filename)
    return toReturn

def prepare_targetFrame() -> pd.DataFrame:
    drugTargets: pd.DataFrame = get_drugTargets()
    drugTargets["DRUG_STANDARD"] = [str(drug).upper().replace(" ","").replace("_", "").replace("(","").replace(")","") for drug in drugTargets["DRUG"].values]
    drugTargets.dropna(axis = "index", subset = "TARGET")
    return drugTargets

def coallate_sc(files: list, dtFrame: pd.DataFrame):
    # Iterate over files of SC dataframes and add the relevant information
    for fileDir in files:
        # Get information about this SC file
        filename = fileDir.split(os.sep)[0]
        measure, method, _ = filename.split("-")
        colStart = f"{measure}-{method}"
        # Read in data
        df = pd.read_csv(fileDir, sep = "\t")
        # Iterate over relevant drugs and genes in dtFrame, and fetch them from the SC data
        toAdd = []
        for drug, gene in zip(dtFrame["DRUG_STANDARD"].values, dtFrame["TARGET"].values):
            if(drug not in df.columms or gene not in df.index):
                toAdd.append((np.nan, np.nan, np.nan))
            toAdd.append((df[drug][gene].values[0], np.mean(df[drug].values), np.std(df[drug].values)))
        newFrame = pd.DataFrame(data = toAdd, columns = [f"{colStart} {m}" for m in ["VALUE", "MEAN", "DEV"]])
        dtFrame = pd.concat([dtFrame, newFrame], axis = "columns")
    return dtFrame

def add_zscore(targetFrame: pd.DataFrame, scData: pd.DataFrame, colTitle: str) -> pd.DataFrame:
    newline = []
    for drug, target in zip(targetFrame["DRUG_STANDARD"].values, targetFrame["TARGET"].values):
        if(drug not in scData.columns or target not in scData.index):
            if(drug not in scData.columns):
                print(f"Error, encountered unknown drug: {drug}")
            if(target not in scData.index):
                print(f"Error, encountered unknown target gene: {target}")
            newline.append(np.nan)
            continue
        rel = scData[drug]
        score = scData.at[target, drug]
        newline.append(np.divide(score - np.mean(rel.values), np.std(rel.values)))
    targetFrame[colTitle] = newline
    return targetFrame



if(__name__=="__main__"):
    main()
