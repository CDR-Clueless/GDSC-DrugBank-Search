#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 23 Jul 2026

@author: jds40
"""

import os
import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd
from matplotlib import pyplot as plt

from typing import Optional, Union, Tuple

DAVID_DIR: str = os.path.join("Data", "Results", "DAVID-Analysis", "Downloads")
OUTPUT_DIR: str = os.path.join("Data", "Results", "DAVID-Analysis")

def main():
    plot_david(saveDir = "", show_graph=False)

def plot_david(saveDir: str = "", show_graph: bool = True):
    fig, ax = plt.subplots(layout='constrained', nrows = 3, ncols=2, figsize = (12.8, 9.6))
    k = 0

    keys = {"Tissue": "tissue", "Interations": "interaction", "Pathways": "pathway", "Protein Domains": "protein", "Transcription Factors": "transcription"}

    statsResults: dict = {}

    for i in range(3):
        for j in range(2):

            if(k>=len(keys)):
                continue

            title = list(keys.keys())[k]
            key = keys[title]
            rel = ax[i][j]

            p, pNon = import_david(targType="predicted", dtype = key), import_david(targType="non-predicted", dtype = key)

            # Combine the two DataFrames into one with 5 Columns: Full Term, Predicted Proportion, Predicted Count, Non-Predicted Proportion, Non-Predicted Count
            combined = p[["Full Term", "Count", "Proportion"]]
            combined = combined.merge(pNon[["Full Term", "Proportion", "Count"]], left_on="Full Term", right_on="Full Term")
            combined.rename(columns={"Full Term": title, "Proportion_x": "Predicted Proportion", "Proportion_y": "Non-Predicted Proportion",
                                     "Count_x": "Predicted Count", "Count_y": "Non-Predicted Count"}, inplace=True)
            toAdd = pNon.loc[~pNon["Full Term"].isin(combined[title])]
            toAdd = toAdd.rename(columns = {"Full Term": title, "Proportion": "Non-Predicted Proportion", "Count": "Non-Predicted Count"})
            toAdd = toAdd[[title, "Non-Predicted Proportion", "Non-Predicted Count"]]
            toAdd["Predicted Proportion"] = 0.0
            toAdd["Predicted Count"] = 0
            combined = pd.concat([combined, toAdd])

            for column, offset in zip(["Predicted Proportion", "Non-Predicted Proportion"], [-0.2, 0.2]):
                if(combined.shape[0]>300):
                    rel.scatter(np.array(range(len(combined))), combined[column], label = column)
                else:
                    rel.bar(np.array(range(len(combined)))+offset, combined[column], width = 0.4, label = column)

            # Labels, titles etc.
            #rel.set_xticks(np.array(range(len(combined))))
            #rel.set_xticklabels(combined[title], ha = "right")
            #rel.tick_params("x", labelrotation=45, labelsize = "x-small")
            rel.set_ylabel("Proportion")
            rel.set_title(f"Target Proportions by {title} type")
            rel.legend(loc="upper right")

            # Calculate chi2 contingency statistic between the two
            valsReal = combined[["Predicted Count", "Non-Predicted Count"]].to_numpy(dtype = int).T
            valsProp = combined[["Predicted Proportion", "Non-Predicted Proportion"]].to_numpy(dtype = float)


            statsResults[title] = {"Counts": chi2_contingency(valsReal).pvalue, "Proportions": chi2_contingency(valsProp).pvalue}
            k += 1

    print(statsResults)

    if(saveDir!=""):
        plt.savefig(saveDir)
    if(show_graph):
        plt.show()
    plt.clf()
    plt.close()
    return

def import_david(dtype: str = "tissue", targType: str = "predicted") -> pd.DataFrame:
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

if(__name__=="__main__"):
    main()
