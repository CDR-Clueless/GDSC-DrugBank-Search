#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 29 Jul 2026

@author: jds40
"""

import os
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import pyarrow.parquet as pq
from sqlite3 import connect

DRUGS_TSV: str = os.path.join("Data", "Results", "Target-Analysis", "Pearson Threshold Labels p < 0.05.tsv")

def main():
    # Load in OpenTargets Data
    dfTarget = pq.read_table(os.path.join("Data", "Raw Data", "OpenTargets", "drug_mechanism_of_action", "part-00000-10b94b1b-f29a-440c-98e0-c91862b6d2a8-c000.snappy.parquet")).to_pandas()
    dfTarget2 = pq.read_table(os.path.join("Data", "Raw Data", "OpenTargets", "drug_mechanism_of_action", "part-00001-10b94b1b-f29a-440c-98e0-c91862b6d2a8-c000.snappy.parquet")).to_pandas()
    dfTarget = pd.concat([dfTarget, dfTarget2], ignore_index=True)

    # Load in ChEMBL Data
    conn = connect(database = os.path.join("Data", "Raw Data", "ChEMBL", "chembl_37", "chembl_37_sqlite", "chembl_37.db"))
    cursor = conn.cursor()
    out = cursor.execute("SELECT CHEMBL_ID, COMPOUND_NAME FROM MOLECULE_DICTIONARY INNER JOIN COMPOUND_RECORDS ON MOLECULE_DICTIONARY.MOLREGNO = COMPOUND_RECORDS.MOLREGNO")
    dfCmbl = pd.DataFrame(data = out.fetchall(), columns = ["ChEMBL ID", "Common Name"])

    # Use ChEMBL data to add common drug names to OpenTargets Data
    dfTarget[["CID1", "CID2"]] = pd.DataFrame(dfTarget.chemblIds.tolist(), index= dfTarget.index)
    mapper = dict(zip(dfCmbl["ChEMBL ID"].values, dfCmbl["Common Name"].values))
    mapper[None] = None
    dfTarget["N1"] = dfTarget["CID1"].map(mapper).str.upper()
    dfTarget["N2"] = dfTarget["CID2"].map(mapper).str.upper()

    # Load in good/bad drug TSV data
    dfDrug = pd.read_csv(DRUGS_TSV, sep = "\t")
    goodDrugsRaw = dfDrug["Drugs With >=1 Target Genes Predicted"].dropna()
    badDrugsRaw = dfDrug["Drugs With No Target Genes Predicted"].dropna()

    goodDrugsCountRaw = len(goodDrugsRaw)
    goodDrugsCount = sum(goodDrugsRaw.isin(dfTarget["N1"]) | goodDrugsRaw.isin(dfTarget["N2"]))
    badDrugsRawCount = len(badDrugsRaw)
    badDrugsCount = sum(badDrugsRaw.isin(dfTarget["N1"]) | badDrugsRaw.isin(dfTarget["N2"]))

    goodDrugs = goodDrugsRaw[goodDrugsRaw.isin(dfTarget["N1"]) | goodDrugsRaw.isin(dfTarget["N2"])].unique()
    badDrugs = badDrugsRaw[badDrugsRaw.isin(dfTarget["N1"]) | badDrugsRaw.isin(dfTarget["N2"])].unique()

    goodTargTypes = dfTarget.loc[dfTarget["N1"].isin(goodDrugs) | dfTarget["N2"].isin(goodDrugs)].targetType
    badTargTypes = dfTarget.loc[dfTarget["N1"].isin(badDrugs) | dfTarget["N2"].isin(badDrugs)].targetType
    goodActionTypes = dfTarget.loc[dfTarget["N1"].isin(goodDrugs) | dfTarget["N2"].isin(goodDrugs)].actionType
    badActionTypes = dfTarget.loc[dfTarget["N1"].isin(badDrugs) | dfTarget["N2"].isin(badDrugs)].actionType

    print(len(goodActionTypes))
    print(len(badActionTypes))

    # Make list of zipped proportions of target types and action types, then plot them
    fig, axs = plt.subplots(nrows = 2, figsize = (12.8, 9.6))
    indexTranslator = {"Target": 0, "Action": 1}
    for aot, gTA, bTA in zip(["Target", "Action"], [goodTargTypes, goodActionTypes], [badTargTypes, badActionTypes]):
        # gTA is the good targets (all), bTA is the bad targets (all), and aot makes clear whether we're dealing with actions or targets
        props, vals = [], []
        # Go through all types in gTA and add proportions for all of them
        for t in gTA.unique():
            gT = len(gTA.loc[gTA==t])
            bT = len(bTA.loc[bTA==t])
            tT = gT + bT
            props.append((t, gT/tT, bT/tT))
            vals.append((t, gT, bT))
        # Go through any types in bTA not in gTA and add the (now known) proportion for them
        for t in bTA.unique():
            if(t not in gTA.unique()):
                props.append((t, 0, len(bTA.loc[bTA==t])))

        # Plot grouped bar chart
        ax = axs[indexTranslator[aot]]
        for label, values, counts, offset in zip(["Good Targets", "Bad Targets"],
                                         [[props[i][1] for i in range(len(props))], [props[i][2] for i in range(len(props))]],
                                         [[vals[i][1] for i in range(len(vals))], [vals[i][2] for i in range(len(vals))]],
                                         [-0.2, 0.2]):
            ax.bar(np.array(range(len(props)))+offset, values, width = 0.4, label = label)
            for i in range(len(props)):
                ax.text(x = i + (offset/2), y = values[i] / 2, s = counts[i], fontsize = "x-small", ha = "center")
        # Labels, titles etc.
        ax.set_xticks(np.array(range(len(props))))
        ax.set_xticklabels([props[i][0] for i in range(len(props))], ha = "right")
        ax.tick_params("x", labelrotation=15, labelsize = "x-small")
        ax.set_ylabel("Proportion")
        ax.set_title(f"{aot} Type Proportions for Drugs with and without predicted Targets")
        ax.legend(loc="upper right")
    plt.show()


    #print(dfTarget[["chemblIds", "CID1", "CID2", "N1", "N2"]])
    return

if(__name__=="__main__"):
    main()
