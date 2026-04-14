#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Apr 2026

@author: jds40
"""

import os
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json

from modality_analysis import get_survivability_threshold
from data_handler import DataHandler

CLEAN_DATA_DIR: str = os.path.join("Data", "Laurence-Data")

def main():
    handler = DataHandler((("AllByAll", os.path.join(CLEAN_DATA_DIR, "AllGenesByAllDrugs.tsv")),
                          ("comboSquared", os.path.join("Data","Results","Survivability-Correlations","GDSCC-Combo eMax-AllDrugsByAllGenes.tsv")),
                          ("leftSquared", os.path.join("Data","Results","Survivability-Correlations","GDSCC-Left Drug eMax-AllDrugsByAllGenes.tsv")),
                          ("rightSquared", os.path.join("Data","Results","Survivability-Correlations","GDSCC-Right Drug eMax-AllDrugsByAllGenes.tsv"))))
    
    handler.datasets["AllByAll"] = handler.datasets["AllByAll"].rename(columns = {"Unnamed: 0": "Drug"})
    for key in ["comboSquared", "leftSquared", "rightSquared"]:
        handler.datasets[key] = handler.datasets[key].set_index("symbol")

    dfc, dfl, dfr = handler.datasets["comboSquared"], handler.datasets["leftSquared"], handler.datasets["rightSquared"]
    with open(os.path.join("Data", "Results", "GDSCC drug-gene correlation frequency histograms", "stats.json"), "r") as f:
        stats = json.load(f)
    
    return

def target_origins(dfc: pd.DataFrame, dfl: pd.DataFrame, dfr: pd.DataFrame, stats: Optional[dict] = None, ) -> None:
    allTargetOrigins: dict = {}
    for i, col in enumerate(dfc.columns):
        sideTargets = {}
        for side, df in zip(["Combo", "Left", "Right"], [dfc, dfl, dfr]):
            thresh = get_survivability_threshold(col, stats, df[col].values, mode = "GDSCC", side = side)
            sideTargets[side] = list((df[col].loc[df[col]>=thresh]).index)
        origins: dict = {"Both": 0, "Left": 0, "Right": 0, "New": 0}
        for target in sideTargets["Combo"]:
            # Check whether this combo target is in the left and/or right lists
            leftTarget, rightTarget = False, False
            if(target in sideTargets["Left"]):
                leftTarget = True
            if(target in sideTargets["Right"]):
                rightTarget = True
            # Increment the appropriate target origins value
            if(leftTarget and rightTarget):
                origins["Both"] += 1
            elif(leftTarget):
                origins["Left"] += 1
            elif(rightTarget):
                origins["Right"] += 1
            else:
                origins["New"] += 1
        allTargetOrigins[col] = deepcopy(origins)
        if(i>4):
            pass
            #break
    
    # Turn dictionary data into usable lists for plotting
    yBoth, yLeft, yRight, yNew = [], [], [], []
    for key in allTargetOrigins.keys():
        rel = allTargetOrigins[key]
        yBoth.append(rel["Both"])
        yLeft.append(rel["Left"])
        yRight.append(rel["Right"])
        yNew.append(rel["New"])
    ## Sort of all of these by total target count
    # Get totals and create an old -> new index list
    totals = [yBoth[i] + yLeft[i] + yRight[i] + yNew[i] for i in range(len(yBoth))]
    indexes = [i for i in range(len(totals))]
    sorter = [ind for _, ind in sorted(zip(totals, indexes))[::-1]]
    # Make new arrays and populate them with sorted versions using the old -> new index list
    # Also sort the totals, because it's useful
    sortedCounts = [np.zeros(len(totals), dtype = int) for _ in range(5)]
    for i, ind in enumerate(sorter):
        for j, counts in enumerate([yBoth, yLeft, yRight, yNew, totals]):
            sortedCounts[j][i] = counts[ind]
    # Replace the old lists
    yBoth, yLeft, yRight, yNew, totals = sortedCounts
    ## Graph creation
    # Set bottom values which increment each time
    bottom = np.zeros(len(yBoth), dtype = int)
    fig, ax = plt.subplots()
    ax.plot([], [], color = "none", linestyle = "-", label = "Target Source")
    for source, count in zip(["Both drugs", "Drug A", "Drug B", "New"], [yBoth, yLeft, yRight, yNew]):
        p = ax.bar(x = allTargetOrigins.keys(), height = count, label = source, bottom = bottom)
        bottom += count
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        'pgf.preamble': r'\\usepackage{amsmath}'
    })

    ax.set_title(r"$GDSC^2$ Combination SCP targets + origins")
    ax.set_xticks(range(len(totals)), labels = ["" for _ in range(len(totals))])
    ax.set_xlabel("Drug Combination")
    ax.set_ylabel("Count")
    
    plt.legend()

    plt.savefig(os.path.join("Data", "Results", "GDSCC Target Analysis", "Target origins.png"))
    
    plt.show()
    plt.close()
    return

if(__name__=="__main__"):
    main()