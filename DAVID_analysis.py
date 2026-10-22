#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 23 Jul 2026

@author: jds40
"""

import os
from copy import deepcopy
import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd
import json
from matplotlib import pyplot as plt
from matplotlib_venn import venn2

from typing import Optional, Union, Tuple

from stats_functions import chisquare_hom

DAVID_DIR: str = os.path.join("Data", "Results", "DAVID-Analysis")
OUTPUT_DIR: str = os.path.join("Data", "Results", "DAVID-Analysis", "Venn AllGenes")

def main():
    if(not os.path.exists(OUTPUT_DIR)):
        os.mkdir(OUTPUT_DIR)
    plot_david_pvals(savedir=OUTPUT_DIR, geneType="all")
    #check_DAVID_original()
    #prep_DAVID_list()
    #update_metadata()
    return

def update_metadata(DAVIDdir: str = DAVID_DIR):
    for gType in ["AllGenes", "BestGenes"]:
        metadata = {}
        folder = f"Downloads-{gType}"
        for predictionType in ["Predicted", "NonPredicted"]:
            predictionTypeFormatted = {"Predicted": "Predicted", "NonPredicted": "Non-Predicted"}[predictionType]
            counts, sources = {}, {}
            # First, get the original number of genes in this category
            origName = "{} {} Genes.csv".format(predictionTypeFormatted,
                                                {"AllGenes": "All", "BestGenes": "Best"}[gType])
            orig = pd.read_csv(os.path.join(DAVIDdir, origName))
            counts["Original"] = orig[orig.columns[0]].drop_duplicates().shape[0]
            # Next, get the number of genes DAVID was able to identiy in this category
            if(gType=="AllGenes"):
                davidName = f"DAVIDGeneListReport_Target Genes {predictionTypeFormatted}.csv"
            else:
                davidName = f"DAVIDGeneListReport_Best Gene for {predictionTypeFormatted} Drug.csv"
            david = pd.read_csv(os.path.join(DAVIDdir, folder, davidName))
            counts["DAVID"] = david["User ID"].drop_duplicates().shape[0]
            # Now, go through each file type in question and get the relevant data
            for fType in ["Disease", "Tissue", "Interactions", "Pathways", "ProteinDomains", "TranscriptionFactors"]:
                fName = f"DAVIDChartReport_{gType}{predictionType}_{fType}.csv"
                df = pd.read_csv(os.path.join(DAVIDdir, folder, fName))
                counts[fType] = int(df["List Total"].values[0])
                sources[fType] = df["Category"].values[0]
            metadata[predictionTypeFormatted] = deepcopy({"Totals": counts, "Sources": sources})
        with open(os.path.join(DAVIDdir, folder, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent = 4)
    return

def check_DAVID_original(DAVIDdir: str = os.path.join(DAVID_DIR, "Downloads-BestGenes"), origDir: str = os.path.join("Data", "Results", "Target-Analysis", "Pearson Threshold Labels p < 0.05.tsv")):
    """Compare the list of genes found by DAVID from the original list to the original list itself to see what's missing

    Args:
        DAVIDdir (str, optional): _description_. Defaults to DAVID_DIR.
        origDir (str, optional): _description_. Defaults to os.path.join("Data", "Results", "Target-Analysis", "Pearson Threshold Labels p < 0.05.tsv").
    """
    # Get DAVID file directories
    npDdir, pDdir = None, None
    for filename in os.listdir(DAVIDdir):
        if("GeneListReport" in filename and "Non-Predicted" in filename):
            npDdir = os.path.join(DAVIDdir, filename)
        elif("GeneListReport" in filename and "Predicted" in filename):
            pDdir = os.path.join(DAVIDdir, filename)

    dfpD, dfnpD, dfO = pd.read_csv(pDdir), pd.read_csv(npDdir), pd.read_csv(origDir, sep = "\t")
    relpD, relnpD, relpO, relnpO = dfpD["User ID"], dfnpD["User ID"], dfO["Target Genes Predicted"].dropna(), dfO["Target Genes Not Predicted"].dropna()

    pdiff, npdiff = relpO.loc[~relpO.isin(relpD)], relnpO.loc[~relnpO.isin(relnpD)]
    print(pdiff)
    print(npdiff)

def prep_DAVID_list(origDir: str = os.path.join("Data", "Results", "Target-Analysis", "Pearson Threshold Labels p < 0.05.tsv"), outDir: str = DAVID_DIR):
    """Make a list to submit to DAVID from the tsv file with information on predicted/non-predicted genes from DepMap data

    Args:
        outDir (DAVID_DIR): _description_
        origDir (str, optional): _description_. Defaults to os.path.join("Data", "Results", "Target-Analysis", "Pearson Threshold Labels p < 0.05.tsv").
    """
    ftype = origDir.split(".")[-1].lower()
    df = pd.read_csv(origDir, sep = {"csv": ",", "tsv": "\t"}[ftype])
    for title, col in zip(["Predicted All Genes", "Non-Predicted All Genes", "Predicted Best Genes", "Non-Predicted Best Genes"],
                           ["Target Genes Predicted", "Target Genes Not Predicted", "Best Gene for Predictable Drug", "Best Gene for non-predictable Drug"]):
        df[col].dropna().drop_duplicates().to_csv(os.path.join(outDir, f"{title}.csv"), sep = ",", lineterminator = "\n", index = False)

def plot_david_pvals(savedir: str = "", geneType: str = "all"):

    # Ensure savedir isn't wrong genetype
    rel = savedir.split(os.sep)[-1]
    if(("all" in rel.lower() and geneType.lower().strip() == "best") or ("best" in rel.lower() and geneType.lower().strip() == "all")):
        if(geneType=="all"):
            rel = rel.replace("Best", "All").replace("best", "all").replace("BEST", "ALL")
        else:
            rel = rel.replace("All", "Best").replace("all", "best").replace("ALL", "BEST")
        print(f"Corrected save directory from {savedir} to {savedir.replace(savedir.split(os.sep)[-1], rel)}")
    savedir = savedir.replace(savedir.split(os.sep)[-1], rel)


    keys = {"Tissue": "tissue", "Interactions": "interaction", "Pathways": "pathway", "Protein Domains": "protein", "Transcription Factors": "transcription",
            "Disease": "disease", "Cancer": "cancer"}

    for key in keys:
        p, pNon = import_david(targType="predicted", dtype = keys[key], geneType=geneType), import_david(targType="non-predicted", dtype = keys[key], geneType=geneType)
        nf = []
        #print(p.columns)
        for term in p["Full Term"].unique():
            if(term in pNon["Full Term"].unique()):
                nf.append((term, p.loc[p["Full Term"]==term]["Benjamini"].values[0], pNon.loc[pNon["Full Term"]==term]["Benjamini"].values[0]))
            else:
                nf.append((term, p.loc[p["Full Term"]==term]["Benjamini"].values[0], np.float64(1.0)))
        for term in pNon["Full Term"].unique():
            if(term not in p["Full Term"].unique()):
                nf.append((term, np.float64(1.0), pNon.loc[pNon["Full Term"]==term]["Benjamini"].values[0]))
        df = pd.DataFrame(data = nf, columns = ["Full Term", "Predicted gene p-value", "Non-predicted gene p-value"])
        df["Predicted p < 0.05"] = df["Predicted gene p-value"] < 0.05
        df["Non-predicted p < 0.05"] = df["Non-predicted gene p-value"] < 0.05
        a, b = set(df.loc[df["Predicted p < 0.05"]==True]["Full Term"].values), set(df.loc[df["Non-predicted p < 0.05"]==True]["Full Term"].values)
        venn = venn2((a, b), set_labels=("Predicted Genes", "Non-Predicted Genes"), set_colors=("g", "r"))
        if(key in ["Tissue", "Cancer"]):
            venn.get_label_by_id('100').set_text('\n'.join(map(str,a-b)))
            venn.get_label_by_id('110').set_text('\n'.join(map(str,a&b)))
            venn.get_label_by_id('010').set_text('\n'.join(map(str,b-a)))
            plt.axis('on')
        plt.title(f"{key} type associations")
        if(savedir != ""):
            plt.savefig(os.path.join(savedir, f"{key}-venn.png"))
            plt.close()
        else:
            plt.show()
    return

def plot_david_proportions(saveDir: str = "", show_graph: bool = True, grouped_graph: bool = False):
    if(grouped_graph):
        fig, ax = plt.subplots(layout='constrained', nrows = 3, ncols=2, figsize = (12.8, 9.6))
    k = 0

    keys = {"Tissue": "tissue", "Interactions": "interaction", "Pathways": "pathway", "Protein Domains": "protein", "Transcription Factors": "transcription"}

    statsResults: dict = {}

    for i in range(3):
        for j in range(2):

            if(k>=len(keys)):
                continue

            if(not grouped_graph):
                fig, ax = plt.subplots()
                rel = ax
                if(saveDir!=""):
                    outDir = os.path.join(saveDir, f"{list(keys.keys())[k]}.png")
                else:
                    outDir = ""
            else:
                outDir = saveDir
                rel = ax[i][j]

            title = list(keys.keys())[k]
            key = keys[title]

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


            statsResults[title] = {"Scipy": chi2_contingency(valsReal).pvalue, "Manual": chisquare_hom(valsReal),
                                    "Manual >=5": chisquare_hom(valsReal, minimum = 5)}

            if(not grouped_graph):
                print(f"{title} chi squared results: {statsResults[title]}")
                if(outDir != ""):
                    plt.savefig(outDir)
                if(show_graph):
                    plt.show()
                plt.close()
            k += 1

    if(grouped_graph):
        print(statsResults)
        if(outDir != ""):
            plt.savefig(outDir)
        if(show_graph):
            plt.show()
        plt.close()
    return

def import_david(dtype: str = "tissue", targType: str = "predicted", geneType: str = "all") -> pd.DataFrame:
    dtype = dtype.lower().strip()
    targType = targType.lower().strip()
    # Determine appropriate file suffix for data type
    dtypes = {"tissue": "Tissue", "interaction": "Interactions", "pathway": "Pathways",
              "protein": "ProteinDomains", "transcription": "TranscriptionFactors",
              "disease": "Disease", "cancer": "Disease"}
    suffix = None
    for key in dtypes:
        if(dtype in key):
            suffix = dtypes[key]
    if(suffix is None):
        print(f"Unusable data type entered: {dtype}.\nValid Data types: {list(dtypes.keys())}")
        return


    # Format gene and target type
    if(geneType.lower().strip()=="all"):
        geneType = "All"
    else:
        geneType = "Best"
    if("non" in targType.lower().strip()):
        targType = "NonPredicted"
    else:
        targType = "Predicted"

    # Load DataFrame, add proportion and return it
    df = pd.read_csv(os.path.join(DAVID_DIR, f"Downloads-{geneType}Genes", f"DAVIDChartReport_{geneType}Genes{targType}_{suffix}.csv"))
    
    df["Proportion"] = df["Count"] / df["List Total"]

    # If cancers specifically are being requested, filter out all non-cancer entries
    if(dtype=="cancer"):
        # First, get list of cancers, and make them all lowercase for better compatibility
        with open(os.path.join(DAVID_DIR, "cancers.txt"), "r") as f:
            cancers = pd.Series(f.read().split("\n")).replace("", np.nan).dropna().drop_duplicates()
        cancers = cancers.str.lower()
        # Make a standard 'term' which is also lowercase
        df["Term Standard"] = df["Term"].str.lower()
        # Only keep entries where the standardised term is in the list of cancers or cancer is in the name, then remove the standardised term column
        df = df.loc[(df["Term Standard"].isin(cancers)) | (df["Term Standard"].str.contains("cancer"))]
        del df["Term Standard"]
    return df

if(__name__=="__main__"):
    main()
