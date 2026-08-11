#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 11 Aug 2026

@author: jds40
"""

import os

import numpy as np
import pandas as pd

from target_functions import get_drugTargets

DEFAULT_FILE_LOC: str = os.path.join("Data", "Results", "Survivability-Correlation", "GDSC")

def main():
    toImport = get_sc_files()
    dTargets = prepare_targetFrame()
    print(dTargets)
    return

def get_sc_files(fileDir: str = DEFAULT_FILE_LOC):
    if(not os.path.exists(fileDir)):
        print(f"Directory not found: {fileDir}")
        return []
    toReturn: list = []
    for filename in os.listdir(fileDir):
        if("alldrugsbyallgenes.tsv" in filename.lower()):
            toReturn.append(os.path.join(fileDir, filename))
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


if(__name__=="__main__"):
    main()
