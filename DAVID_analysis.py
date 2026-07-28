#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 23 Jul 2026

@author: jds40
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from typing import Optional, Union, Tuple

DAVID_DIR: str = os.path.join("Data", "Results", "DAVID-Analysis", "Downloads")
OUTPUT_DIR: str = os.path.join("Data", "Results", "DAVID-Analysis")

def main():
    dtypes = {"tissue": "Tissues", "interaction": "Interactions", "pathway": "Pathways",
              "protein": "ProteinDomains", "transcription": "TranscriptionFactors"}
    fig, ax = plt.subplots(layout='constrained', nrows = 3, ncols=2, figsize = (12.8, 9.6))
    k = 0

    keys = {"Tissue": "tissue", "Interations": "interaction", "Pathways": "pathway", "Protein Domains": "protein", "Transcription Factors": "transcription"}

    for i in range(3):
        for j in range(2):

            if(k>=len(keys)):
                continue

            title = list(keys.keys())[k]
            key = keys[title]
            rel = ax[i][j]

            p, pNon = import_david(targType="predicted", dtype = key), import_david(targType="non-predicted", dtype = key)

            # Combine the two DataFrames into one
            combined = p[["Full Term", "Proportion"]]
            combined = combined.merge(pNon[["Full Term", "Proportion"]], left_on="Full Term", right_on="Full Term")
            combined.rename(columns={"Full Term": title, "Proportion_x": "Predicted Proportion", "Proportion_y": "Non-Predicted Proportion"}, inplace=True)
            toAdd = pNon.loc[~pNon["Full Term"].isin(combined[title])]
            toAdd.rename(columns = {"Full Term": title, "Proportion": "Non-Predicted Proportion"}, inplace=True)
            toAdd = toAdd[[title, "Non-Predicted Proportion"]]
            toAdd["Predicted Proportion"] = [0.0 for _ in range(len(toAdd))]
            combined = pd.concat([combined, toAdd])

            for column, offset in zip(["Predicted Proportion", "Non-Predicted Proportion"], [-0.2, 0.2]):
                rel.bar(np.array(range(len(combined)))+offset, combined[column], width = 0.4, label = column)

            # Labels, titles etc.
            #rel.set_xticks(np.array(range(len(combined))))
            #rel.set_xticklabels(combined[title], ha = "right")
            #rel.tick_params("x", labelrotation=45, labelsize = "x-small")
            rel.set_ylabel("Proportion")
            rel.set_title(f"Target Proportions by {title} type")
            rel.legend(loc="upper right")

            k += 1

    plt.savefig("Test DAVID Output.png")


    return

def import_david(dtype: str = "tissue", targType: str = "predicted"):
    dtype = dtype.lower().strip()
    targType = targType.lower().strip()
    # Determine appropriate file suffix for data type
    dtypes = {"tissue": "Tissues", "interaction": "Interactions", "pathway": "Pathways",
              "protein": "ProteinDomains", "transcription": "TranscriptionFactors"}
    suffix = None
    for key in dtypes:
        if(dtype in key):
            suffix = dtypes[key]
    if(suffix is None):
        print(f"Unusable data type entered: {dtype}.\nValid Data types: {list(dtypes.keys())}")
        return

    # Determine whether predicted or non-predicted targets are desired
    if("non" in targType):
        targP = "Non-Predicted"
    else:
        targP = "Predicted"

    # Load DataFrame and 
    df = pd.read_csv(os.path.join(DAVID_DIR, f"DAVIDChartReport_{targP} Targets_{suffix}.csv"))
    df["Proportion"] = df["Count"] / df["List Total"]
    return df

def import_tissue() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(os.path.join(DAVID_DIR, "DAVIDChartReport_Non-Predicted Targets_Tissues.csv")), pd.read_csv(os.path.join(DAVID_DIR, "DAVIDChartReport_Predicted Targets_Tissues.csv"))



if(__name__=="__main__"):
    main()
