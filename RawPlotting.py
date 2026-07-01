#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 30 Jun 2026

@author: jds40
"""

DEBUG_MODE: bool = True

import os
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, linregress

from matplotlib import pyplot as plt
from tqdm import tqdm
import json

CLEANED_DATA_DIR: str = os.path.join("Data", "Laurence-Data")
DEFAULT_CRISPR_FILE: str = os.path.join(CLEANED_DATA_DIR,"CRISPRGeneDependency.csv")
DEFAULT_HUGO_FILE: str = os.path.join(CLEANED_DATA_DIR, "hgnc_complete_set.tsv")
DEFAULT_CELL_INFO_FILE: str = os.path.join(CLEANED_DATA_DIR, "Model.csv")
DEFAULT_DRUG1_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC1_drug_results_target_cleaned7.tsv')
DEFAULT_DRUG2_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC2_drug_results_target_cleaned7.tsv')
DEFAULT_DRUG_COMB_FILE: str = os.path.join("Data", "Raw Data", "GDSCC")
DEFAULT_OUTPUT_DIR: str = os.path.join("Data", "Results", "Response-Essentiality Plots")

from target_analysis import get_target_list

DEBUG_MODE: bool
localLow = os.path.join("Local", "localVars.json")
if(os.path.exists(localLow)):
    with open(localLow, "rb") as f:
        DEBUG_MODE = bool(json.load(f)["DEBUG_MODE"])
else:
    DEBUG_MODE = False

def main():
    targetDict = get_target_list()
    di = 0
    for drug in tqdm(targetDict.keys(), desc = "Finding Drug SC graphs"):
        plot_graphs([drug], targetDict[drug])
        di += 1
        if(DEBUG_MODE and di>20):
            return

def plot_graphs(drugs: list, genes: list):
    # Format choices of drugs and genes
    drugs = [str(drug).upper().strip() for drug in drugs]
    genes = [str(gene).upper().strip() for gene in genes]
    if(len(drugs)<1 or len(genes)<1):
        return
    # Compile dictionary of relevant file locations
    fileLocs = {}
    for i, name in enumerate(["crispr", "hugo", "cellinfo", "gdsc1", "gdsc2"]):
        fileLocs[name] = [DEFAULT_CRISPR_FILE, DEFAULT_HUGO_FILE, DEFAULT_CELL_INFO_FILE, DEFAULT_DRUG1_FILE, DEFAULT_DRUG2_FILE][i]

    # Get number of CPU's to use for multiprocessing
    cpu_count = max(1, mp.cpu_count()-2)

    # Ensure Output directory exists
    for cdir in [DEFAULT_OUTPUT_DIR, os.path.join(DEFAULT_OUTPUT_DIR, "GDSC1"), os.path.join(DEFAULT_OUTPUT_DIR, "GDSC2")]:
        if(not os.path.exists(cdir)):
            os.mkdir(cdir)

    # Get known CRISPR cell line-gene dependencies (row index = model ID/cell line ID, column = Gene)
    crisprDeps = pd.read_csv(fileLocs["crispr"]).fillna(0.0)

    crisprDeps.rename(columns = {'Unnamed: 0':'ModelID'},inplace=True)
    crisprDeps.set_index('ModelID', inplace=True)
    
    # edit header names to remove spaces etc.
    gg = dict(zip(list(crisprDeps.columns), [i.strip().split()[0]
              for i in list(crisprDeps.columns)]))
    crisprDeps.rename(columns=gg, inplace=True)

    # Get HUGO standardised gene name information
    hgnc = pd.read_table(fileLocs["hugo"], low_memory=False).fillna('')
    hgnc = hgnc[['symbol', 'ensembl_gene_id',
                 'prev_symbol', 'location', 'location_sortable']]
    hgnc.set_index('symbol', inplace=True)
    
    # Correct legacy gene names in crisprDeps using HUGO table
    old_names = set(crisprDeps.columns) & (set(hgnc.index) ^ set(crisprDeps.columns))
    for g_old in old_names:
        g_new = hgnc[hgnc['prev_symbol'].str.contains(g_old)].reset_index()['symbol']
        if len(g_new) == 0 or (g_new[0] not in hgnc.index):
            continue
        else:
            crisprDeps.rename(columns={g_old: g_new[0]}, inplace=True)
    
    # Load in cell line information - useful for linking cell line names/IDs/etc. from other DataFrames to other information from other DataFrames
    clInfo = pd.read_csv(fileLocs["cellinfo"], low_memory=False).fillna('')
    clInfo['OncotreeLineage'] = [x.upper() for x in clInfo['OncotreeLineage']]
    clInfo["OncotreePrimaryDisease"] = clInfo["OncotreePrimaryDisease"].str.replace(' ','_')

    # Separate out cancer types from cell line information as it is of particular note
    cancer_types = set(clInfo['OncotreeLineage'])

    # Load in GDSC data
    drug1 = pd.read_table(fileLocs["gdsc1"], low_memory=False).fillna('')
    drug1["DRUG_NAME"] = drug1["DRUG_NAME"].apply(lambda x:x.upper())
    
    drug2 = pd.read_table(fileLocs["gdsc2"], low_memory=False).fillna('')
    drug2["DRUG_NAME"] = drug2["DRUG_NAME"].apply(lambda x:x.upper())

    # Initilise drug by gene data
    dList = sorted(set(drug1["DRUG_NAME"]) | set(drug2["DRUG_NAME"]))
    drugsRefined = sorted(set(dList) & set(drugs))
    drugsFailed = [drug for drug in drugs if drug not in dList]
    with open(os.path.join(DEFAULT_OUTPUT_DIR, "failedDrugs.txt"), "a+") as f:
        f.write(f"Failed to find drug in GDSC1/2: {','.join(drugsFailed)}")
    if(DEBUG_MODE and len(drugsRefined)>20):
        drugsRefined = drugsRefined[:20]
    drug1.set_index("DRUG_NAME", inplace = True)
    drug2.set_index("DRUG_NAME", inplace = True)
    drug1["pIC50"] = np.multiply(-1, drug1["LN_IC50"])
    drug2["pIC50"] = np.multiply(-1, drug2["LN_IC50"])

    # loop through all indexes, i.e. drugs/compounds, calculating r for all genes
    for di, d in enumerate(drugsRefined):
        for gdscv in ["GDSC1", "GDSC2"]:
            # Get relevant dataset
            rel = {"GDSC1": drug1, "GDSC2": drug2}[gdscv]

            # Ensure necessary directories exist; skip this dataset if the drug is not in it
            if(not os.path.exists(os.path.join(DEFAULT_OUTPUT_DIR, gdscv, d)) and d in rel.index):
                os.mkdir(os.path.join(DEFAULT_OUTPUT_DIR, gdscv, d))
            if(d not in rel.index):
                continue
        
            # Get all available cell lines
            cs = rel[rel.index==d]
            
            # Go through all genes from CRISPR dependencies DataFrame
            genesAvail = crisprDeps.columns
            genesRefined = sorted(set(genesAvail) & set(genes))
            genesFailed = [gene for gene in genes if gene not in genesAvail]
            if(len(genesFailed)>0):
                with open(os.path.join(DEFAULT_OUTPUT_DIR, gdscv, d, "failures.txt"), "a+") as f:
                    f.write(f"Failed to find genes in GDSC1/2: {genesFailed}")
            # Go through target genes and generate plots
            for i, gn in enumerate(genesRefined):
                
                # get dependencies (deps) for all available cell lines, as well as a list of cell lines which
                # were found within the deps DataFrame

                deps = crisprDeps[crisprDeps.index.isin(cs["ModelID"].values)][gn].reset_index()
                if(len(deps)>0):
                    dep_names = deps["ModelID"]
                else:
                    dep_names = None

                # Get DataFrames of whatever is being used for establishing correlations (i.e. pKi, IC50 or eMax values)
                response = cs[(cs.index == d) & 
                            (cs["ModelID"].isin(dep_names))].drop_duplicates \
                            (subset=["ModelID"], keep="first")[["pIC50","ModelID"]]
                mID, ess = deps["ModelID"].values, deps[gn].values
                response["Essentiality"] = response["ModelID"].map({mID[i]: ess[i] for i in range(len(deps))})
                #print(f"Drug: {d}, Gene: {gn}\ncs:\n{cs}\ndeps:\n{deps}\ndep_names:\n{dep_names}\nresponse:\n{response}")
                # Make graphs for GDSC1 and GDSC2
                if(response is None):
                    continue
                plt.scatter(response["Essentiality"], response["pIC50"])
                plt.xlabel(f"{gn} Gene Essentiality")
                plt.ylabel("pIC50 Response")
                plt.title(f"{gdscv} {d} responses against {gn} essentiality")

                # Get Pearson Correlation and line of best fit
                result = linregress(response["Essentiality"].values, response["pIC50"].values)
                m, c, pr, pp, mErr, cErr = result.slope, result.intercept, result.rvalue, result.pvalue, result.stderr, result.intercept_stderr
                plt.plot([0, max(response["Essentiality"])], [c, (m*max(response["Essentiality"]))+c], linestyle = "--", color = "black")
                plt.text(max(response["Essentiality"])-0.05, (m*(max(response["Essentiality"])-.05)+c+((max(response["pIC50"])-min(response["pIC50"]))*.05)), f"Pr = {pr:.2f}")
                plt.savefig(os.path.join(DEFAULT_OUTPUT_DIR, gdscv, d, f"{gn}.png"))
                plt.clf()
                plt.close()
        # If in debug mode, stop after the first 20 drugs
        if(DEBUG_MODE and di>20):
            break
    return

if(__name__=="__main__"):
    main()