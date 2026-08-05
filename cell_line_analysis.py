#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 4 Aug 2026

@author: jds40
"""

import os
from copy import deepcopy
import multiprocessing as mp

import random
import numpy as np
import pandas as pd
import json

from typing import Union
from scipy.stats import linregress
from tqdm import tqdm

if(os.path.exists(os.path.join("Local", "localVars.json"))):
    with open(os.path.join("Local", "localVars.json"), "r") as f:
        DEBUG_MODE: bool = json.load(f)["DEBUG_MODE"]
else:
    DEBUG_MODE: bool = False

SC_DIR: str = os.path.join("Data", "Results", "Survivability-Correlations")

CLEANED_DATA_DIR: str = os.path.join("Data", "Laurence-Data")
DEFAULT_CRISPR_FILE: str = os.path.join(CLEANED_DATA_DIR,"CRISPRGeneDependency.csv")
DEFAULT_HUGO_FILE: str = os.path.join(CLEANED_DATA_DIR, "hgnc_complete_set.tsv")
DEFAULT_CELL_INFO_FILE: str = os.path.join(CLEANED_DATA_DIR, "Model.csv")
DEFAULT_DRUG1_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC1_drug_results_target_cleaned7.tsv')
DEFAULT_DRUG2_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC2_drug_results_target_cleaned7.tsv')
DEFAULT_DRUG_COMB_FILE: str = os.path.join("Data", "Raw Data", "GDSCC")
DEFAULT_OUTPUT_DIR: str = os.path.join("Data", "Results", "Cell-Line-Distances")

def main():
    # Get number of CPU's to use for multiprocessing
    cpu_count = max(1, mp.cpu_count()-2)
    # First, fetch the DataFrame List of Cell Line - Drug Response values
    drugs, genes, cls = sorted(get_drugs()), sorted(get_genes()), sorted(get_cellLines())
    if(DEBUG_MODE):
        drugs, genes, cls = drugs[:2], genes[:2], cls[:50]
    # Get split list of drugs for parallel worker to handle
    toSub = split_list(drugs, cpu_count)
    # Calculate distances between cell lines and linear lines of best fit in parallel, saving results to be coallated later
    starDir = os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap")
    if(not os.path.exists(starDir)):
        os.mkdir(starDir)
    results = mp.Pool(cpu_count).starmap_async(cellWorker,
                                               [(toSub[i],genes, starDir)
                                                               for i in range(cpu_count)]).get()
    # Coallate results
    coallated: dict = {}
    for drug in drugs:
        with open(os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap", f"{drug}.json"), "r") as f:
            drugDict = json.load(f)
        for gene in drugDict:
            for cL in drugDict[gene]:
                if(cL not in coallated):
                    coallated[cL] = {drug: {gene: np.nan for gene in genes} for drug in drugs}
                coallated[cL][drug][gene] = drugDict[gene][cL]
    # Save coallated results and delete temporary starmap store
    for cL in coallated:
        with open(os.path.join(DEFAULT_OUTPUT_DIR, f"{cL}-Distances.json"), "w") as f:
            json.dump(coallated[cL], f)
    for filename in os.listdir(starDir):
        os.remove(os.path.join(starDir, filename))
    os.rmdir(starDir)
    return

# Starmap worker - saves cell line data
def cellWorker(drugs: list, genes: list, outDir: str = os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap")):
    # Iterate over drugs
    for drug in drugs:
        # Check if this drug has already had its results calculated; skip if so
        if(os.path.exists(os.path.join(outDir, f"{drug}.json"))):
            continue
        # Create dictionary to store output calculations
        out = {gene: {} for gene in genes}
        # Iterate over responses for each gdrug-gene pair
        for gene in genes:
            response = load_response(drug, gene)
            # Get Pearson Correlation and line of best fit
            result = linregress(response["Essentiality"].values, response["pIC50"].values)
            m, c, pr, pp, mErr, cErr = result.slope, result.intercept, result.rvalue, result.pvalue, result.stderr, result.intercept_stderr
            # Get distance of every Cell Line available from the curve
            for cL in response["ModelID"].values:
                rel = response.loc[response["ModelID"]==cL][["Essentiality", "pIC50"]].values
                x, y = rel[0][0], rel[0][1]
                # Calculate distance using d = |Ax + By + C| / sqrt(A^2 + B^2)
                # Here A is the slope (variable m), B is -1 (slope and intercept fit y = mx + c, so 0 = mx + c - y therefore B is -1), C is the intercept (variable c)
                # x is the Cell Line response x-coordinate (Essentiality), and y the y-coordinate (pIC50)
                num = (m * x) + (-1 * y) + c
                den = np.sqrt(np.power(m, 2) + 1)
                out[gene][cL] = np.divide(num, den)
        # Save calculations for this drug
        with open(os.path.join(outDir, f"{drug}.json"), "w") as f:
            json.dump(out, f, indent = 4)
    return

def get_drugs() -> tuple:
    # Compile dictionary of relevant file locations
    fileLocs = {}
    for i, name in enumerate(["crispr", "hugo", "cellinfo", "gdsc1", "gdsc2"]):
        fileLocs[name] = [DEFAULT_CRISPR_FILE, DEFAULT_HUGO_FILE, DEFAULT_CELL_INFO_FILE, DEFAULT_DRUG1_FILE, DEFAULT_DRUG2_FILE][i]
    ## Get list of available drugs
    # Load in GDSC data
    drug1 = pd.read_table(fileLocs["gdsc1"], low_memory=False).fillna('')
    drug1["DRUG_NAME"] = drug1["DRUG_NAME"].apply(lambda x:x.upper())
    
    drug2 = pd.read_table(fileLocs["gdsc2"], low_memory=False).fillna('')
    drug2["DRUG_NAME"] = drug2["DRUG_NAME"].apply(lambda x:x.upper())

    dList = sorted(set(drug1["DRUG_NAME"]) | set(drug2["DRUG_NAME"]))
    return tuple(dList)

def get_genes() -> tuple:
    # Compile dictionary of relevant file locations
    fileLocs = {}
    for i, name in enumerate(["crispr", "hugo", "cellinfo", "gdsc1", "gdsc2"]):
        fileLocs[name] = [DEFAULT_CRISPR_FILE, DEFAULT_HUGO_FILE, DEFAULT_CELL_INFO_FILE, DEFAULT_DRUG1_FILE, DEFAULT_DRUG2_FILE][i]
    ## Get list of available genes
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
    return tuple(crisprDeps.columns)

def get_cellLines() -> tuple:
    # Compile dictionary of relevant file locations
    fileLocs = {}
    for i, name in enumerate(["crispr", "hugo", "cellinfo", "gdsc1", "gdsc2"]):
        fileLocs[name] = [DEFAULT_CRISPR_FILE, DEFAULT_HUGO_FILE, DEFAULT_CELL_INFO_FILE, DEFAULT_DRUG1_FILE, DEFAULT_DRUG2_FILE][i]

    # Load in GDSC data
    drug1 = pd.read_table(fileLocs["gdsc1"], low_memory=False).fillna('')
    drug1["DRUG_NAME"] = drug1["DRUG_NAME"].apply(lambda x:x.upper())
    
    drug2 = pd.read_table(fileLocs["gdsc2"], low_memory=False).fillna('')
    drug2["DRUG_NAME"] = drug2["DRUG_NAME"].apply(lambda x:x.upper())

    cList = sorted(set(drug1["ModelID"]) | set(drug2["ModelID"]))

    return cList


def load_response(drug: str, gene: str) -> pd.DataFrame:
    # Compile dictionary of relevant file locations
    fileLocs = {}
    for i, name in enumerate(["crispr", "hugo", "cellinfo", "gdsc1", "gdsc2"]):
        fileLocs[name] = [DEFAULT_CRISPR_FILE, DEFAULT_HUGO_FILE, DEFAULT_CELL_INFO_FILE, DEFAULT_DRUG1_FILE, DEFAULT_DRUG2_FILE][i]

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
    if(DEBUG_MODE and len(dList)>20):
        dList = dList[:20]
    drug1.set_index("DRUG_NAME", inplace = True)
    drug2.set_index("DRUG_NAME", inplace = True)
    drug1["pIC50"] = np.multiply(-1, drug1["LN_IC50"])
    drug2["pIC50"] = np.multiply(-1, drug2["LN_IC50"])

    # loop through all indexes, i.e. drugs/compounds, calculating r for all genes
    for gdscv in ["GDSC2", "GDSC1"]:
        # Get relevant dataset
        rel = {"GDSC1": drug1, "GDSC2": drug2}[gdscv]

        if(drug not in rel.index or gene not in crisprDeps.columns):
            continue
    
        # Get all available cell lines
        cs = rel[rel.index==drug]
            
        # get dependencies (deps) for all available cell lines, as well as a list of cell lines which
        # were found within the deps DataFrame

        deps = crisprDeps[crisprDeps.index.isin(cs["ModelID"].values)][gene].reset_index()
        if(len(deps)>0):
            dep_names = deps["ModelID"]
        else:
            dep_names = None

        # Get DataFrames of whatever is being used for establishing correlations (i.e. pKi, IC50 or eMax values)
        response = cs[(cs.index == drug) & 
                    (cs["ModelID"].isin(dep_names))].drop_duplicates \
                    (subset=["ModelID"], keep="first")[["pIC50","ModelID"]]
        mID, ess = deps["ModelID"].values, deps[gene].values
        response["Essentiality"] = response["ModelID"].map({mID[i]: ess[i] for i in range(len(deps))})
        # Load data into relevant dicionary if possible
        if(response is None):
            continue
        else:
            return response
    return None

def split_list(l: list, parts: int, shuffle: bool = False) -> list:
    """Split a larger list into a given number of component lists (used here for more efficient multiprocessing batches)

    Args:
        l (list): A list
        parts (int): Number of component lists to break the larger list into
        shuffle (bool): Whether the components of the lists should be shuffled (Defaults to False; not shuffling is also more computationally efficient)

    Returns:
        list: List of output lists all of equal size
    """
    # Number of parts to break this into; if we want to split it into more parts than there are, we'll need to add blank lists
    n = min(parts, max(len(l),1))
    if(shuffle == False):
        n = min(parts, max(len(l),1))
        k, m = divmod(len(l), n)
        output = [l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
        # Add extra empty lists if needed
        while(n<parts):
            output.append([])
            n += 1
    else:
        # Shuffle which parts go where
        output = [[] for _ in range(n)]
        choices = [i for i in range(n)]
        for item in l:
            # Decide which output sub-list to put this item in
            choice = random.choice(choices)
            output[choice].append(item)
            # If the sub-list is now longer than the length of the original list divided by number of lists, remove that sub-list as an option for future choices
            if(len(output[choice]) > len(l) / n):
                choices.remove(choice)
    return output


if(__name__=="__main__"):
    main()
