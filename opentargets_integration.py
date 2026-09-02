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

from scipy.stats import chi2_contingency
from stats_functions import cohenD
from scipy.stats import ttest_ind

import pyarrow.parquet as pq
from sqlite3 import connect

DRUGS_TSV: str = os.path.join("Data", "Results", "Target-Analysis", "Pearson Threshold Labels p < 0.05.tsv")

def main():
    #plot_actions()
    plot_pChEMBL()

def plot_pChEMBL():
    # Load in ChEMBL Data
    conn = connect(database = os.path.join("Data", "Raw Data", "ChEMBL", "chembl_37", "chembl_37_sqlite", "chembl_37.db"))
    cursor = conn.cursor()
    out = cursor.execute("SELECT COMPOUND_NAME, PCHEMBL_VALUE FROM COMPOUND_RECORDS INNER JOIN ACTIVITIES ON COMPOUND_RECORDS.RECORD_ID = ACTIVITIES.RECORD_ID")
    dfCmbl = pd.DataFrame(data = out.fetchall(), columns = ["Common Name", "pChEMBL"])
    dfCmbl.dropna(axis = "index", inplace=True)
    dfCmbl["Common Name"] = dfCmbl["Common Name"].str.upper()

    # Load in good/bad drug target data
    dfDrug = pd.read_csv(DRUGS_TSV, sep = "\t")
    goodDrugsRaw = dfDrug["Drugs With >=1 Target Genes Predicted"].dropna()
    badDrugsRaw = dfDrug["Drugs With No Target Genes Predicted"].dropna()

    goodDrugsCountRaw = goodDrugsRaw.shape[0]
    badDrugsRawCount = badDrugsRaw.shape[0]
    goodDrugs = goodDrugsRaw.loc[goodDrugsRaw.isin(dfCmbl["Common Name"])]
    badDrugs = badDrugsRaw.loc[badDrugsRaw.isin(dfCmbl["Common Name"])]
    goodDrugsCount = goodDrugs.shape[0]
    badDrugsCount = badDrugs.shape[0]

    print(f"{str(goodDrugsCountRaw).ljust(3)} -> {str(goodDrugsCount).ljust(3)}")
    print(f"{str(badDrugsRawCount).ljust(3)} -> {str(badDrugsCount).ljust(3)}")

    goodResults = []
    badResults = []

    for drug in goodDrugs:
        rel = dfCmbl.loc[dfCmbl["Common Name"] == drug]
        goodResults.append(np.mean(rel["pChEMBL"].values))
    for drug in badDrugs:
        rel = dfCmbl.loc[dfCmbl["Common Name"] == drug]
        badResults.append(np.mean(rel["pChEMBL"].values))

    # Make plot
    plt.figure(figsize = (12.8, 9.6))
    plt.boxplot(x = [goodResults, badResults], whis = (0.05, 0.95))
    plt.ylabel("Average Drug pChEMBL Value")
    plt.xlabel("Category")
    plt.xticks([1, 2], labels = ["w. Predictable Targets", "wo. Predictable Targets"], rotation = 15)
    plt.title("Average pChEMBL values of drugs with predictable and non-predictable targets")
    # Calculate stats
    tRes = ttest_ind(goodResults, badResults)
    cohen = cohenD((goodResults, badResults))
    print(f"Student t-test result: {tRes.pvalue}\nCohen's D: {cohen}")
    plt.show()
    return

def plot_actions():
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
    badDrugsCountRaw = len(badDrugsRaw)
    badDrugsCount = sum(badDrugsRaw.isin(dfTarget["N1"]) | badDrugsRaw.isin(dfTarget["N2"]))
    goodDrugs = goodDrugsRaw[goodDrugsRaw.isin(dfTarget["N1"]) | goodDrugsRaw.isin(dfTarget["N2"])].unique()
    badDrugs = badDrugsRaw[badDrugsRaw.isin(dfTarget["N1"]) | badDrugsRaw.isin(dfTarget["N2"])].unique()
    goodDrugsCountUnique = len(goodDrugs)
    badDrugsCountUnique = len(badDrugs)

    print(f"Good Drugs: {goodDrugsCountRaw} -> {goodDrugsCount} -> {goodDrugsCountUnique}")
    print(f"Bad Drugs:  {badDrugsCountRaw} -> {badDrugsCount} -> {badDrugsCountUnique}")

    # Set up dictionaries to record target and action types for each 'good'/'bad' drug
    goodTargTypes, goodActionTypes = {tType: 0.0 for tType in dfTarget["targetType"].unique()}, {aType: 0 for aType in dfTarget["actionType"].unique()}
    badTargTypes, badActionTypes = deepcopy(goodTargTypes), deepcopy(goodActionTypes)
    # Record each action and target types associated with each drug of interest
    for drug in goodDrugs:
        rel = dfTarget.loc[(dfTarget.N1 == drug) | (dfTarget.N2 == drug)]
        for aType in rel["actionType"].unique():
            goodActionTypes[aType] += 1.
        for tType in rel["targetType"].unique():
            goodTargTypes[tType] += 1.
    for drug in badDrugs:
        rel = dfTarget.loc[(dfTarget.N1 == drug) | (dfTarget.N2 == drug)]
        for aType in rel["actionType"].unique():
            badActionTypes[aType] += 1.
        for tType in rel["targetType"].unique():
            badTargTypes[tType] += 1.
    # Convert all these counts to proportions of the total number of drugs in each list
    rawCounts: dict = {}
    for total, d, label in zip([goodDrugsCountUnique, goodDrugsCountUnique, badDrugsCountUnique, badDrugsCountUnique],
                        [goodTargTypes, goodActionTypes, badTargTypes, badActionTypes],
                        ["Good Target Types", "Good Action Types", "Bad Target Types", "Bad Action Types"]):
        rawCounts[label] = {}
        for key in d:
            rawCounts[label][key] = d[key]
            d[key] /= total

    for key in deepcopy(list(goodTargTypes.keys())):
        if(goodTargTypes[key]==0.0 and badTargTypes[key]==0.0):
            del goodTargTypes[key]
            del badTargTypes[key]
    for key in deepcopy(list(goodActionTypes.keys())):
        if(goodActionTypes[key]==0.0 and badActionTypes[key]==0.0):
            del goodActionTypes[key]
            del badActionTypes[key]

    # Make list of zipped proportions of target types and action types, then plot them
    fig, axs = plt.subplots(nrows = 2, figsize = (12.8, 9.6))
    indexTranslator, chiRes = {"Target": 0, "Action": 1}, {}
    for aot, gTA, bTA in zip(["Target", "Action"], [goodTargTypes, goodActionTypes], [badTargTypes, badActionTypes]):
        # gTA is the good targets (all), bTA is the bad targets (all), and aot makes clear whether we're dealing with actions or targets

        # Plot grouped bar chart
        ax = axs[indexTranslator[aot]]
        for label, xs, ys, offset in zip(["Drugs w Predictable Targets", "Drugs w/o Predictable Targets"],
                                         [list(gTA.keys()), list(bTA.keys())],
                                         [list(gTA.values()), list(bTA.values())],
                                         [-0.2, 0.2]):
            ax.bar(np.array(range(len(xs)))+offset, ys, width = 0.4, label = label)
            for i in range(len(xs)):
                ax.text(x = i + (offset/2), y = ys[i] / 2, s = f"{ys[i]:.2f}", fontsize = "x-small", ha = "center")
        # Labels, titles etc.
        ax.set_xticks(np.array(range(len(xs))))
        ax.set_xticklabels(xs, ha = "right")
        ax.tick_params("x", labelrotation=15, labelsize = "x-small")
        ax.set_ylabel("Proportion")
        ax.set_title(f"{aot} Type Proportions for Drugs with and without predicted Targets")
        ax.legend(loc="upper right")

        ## Calculate chi square results
        # Get raw counts
        if(aot=="Target"):
            good, bad = rawCounts["Good Target Types"], rawCounts["Bad Target Types"]
        else:
            good, bad = rawCounts["Good Action Types"], rawCounts["Bad Action Types"]
        # Form matrix
        mat = []
        for key in good.keys():
            if(key in bad.keys()):
                mat.append((good[key], bad[key]))
            else:
                mat.append((good[key], 0))
        for key in bad.keys():
            if(key not in good.keys()):
                mat.append((0, bad[key]))
        mat = np.array(mat, dtype = int)
        # Correct to remove rows in which both values are 0
        cMat = mat[np.logical_or(mat[:,0] > 0, mat[:,1] > 0)]
        resRaw, res5 = chi2_contingency(cMat).pvalue, chi2_contingency(cMat[np.logical_or(cMat[:,0] >= 5, cMat[:,1] >= 5)]).pvalue
        chiRes[aot] = {"All": resRaw, "Super-5": res5}
    print(chiRes)
    plt.show()


    return

if(__name__=="__main__"):
    main()
