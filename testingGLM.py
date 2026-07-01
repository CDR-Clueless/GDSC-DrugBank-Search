#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 11 Jun 2026

@author: jds40
"""
import os
from typing import Optional
import multiprocessing as mp

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import json

from drug_search import get_data
from logger import Logger

CLEANED_DATA_DIR: str = os.path.join("Data", "Laurence-Data")
DEFAULT_CRISPR_FILE: str = os.path.join(CLEANED_DATA_DIR,"CRISPRGeneDependency.csv")
DEFAULT_HUGO_FILE: str = os.path.join(CLEANED_DATA_DIR, "hgnc_complete_set.tsv")
DEFAULT_CELL_INFO_FILE: str = os.path.join(CLEANED_DATA_DIR, "Model.csv")
DEFAULT_DRUG1_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC1_drug_results_target_cleaned7.tsv')
DEFAULT_DRUG2_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC2_drug_results_target_cleaned7.tsv')
DEFAULT_DRUG_COMB_FILE: str = os.path.join("Data", "Raw Data", "GDSCC")

# Decide whether this should be a debug mode
if(not os.path.exists(os.path.join("Local", "localVars.json"))):
    DEBUG_MODE: bool = False
else:
    with open(os.path.join("Local", "localVars.json"), "rb") as f:
        DEBUG_MODE: bool = json.load(f)["DEBUG_MODE"]

def dummyLinear(x1: float = 1.0, c: float = 0.1, noise: float = 0.1, lower: float = 0.0, upper: float = 1.0, n: int = 100) -> np.ndarray:
    return (np.linspace(lower, upper, n) * x1) + c + (noise * np.random.uniform(0, 1, n))

def dummyTwo(x1: float = 1.0, x2: float = 1.0, c: float = 0.1, noise: float = 0.1, lower: float = 0.0, upper: float = 1.0, n: int = 100) -> np.ndarray:
    base = np.linspace(lower, upper, n)
    return (base * x1) + (x2 * np.power(base, 2.)) + c + (noise * np.random.uniform(lower, upper, n))

def dummyX(xs: list = [1, 2], c: float = 0.1, noise: float = 0.1, lower: float = 0.0, upper: float = 1.0, n: int = 100) -> tuple[np.ndarray]:
    base = np.linspace(lower, upper, n)
    y = (np.random.uniform(0, 1, n) * noise) + c
    for power, x in enumerate(xs):
        y += x * np.power(base, power+1)
    return np.vstack([np.ones(n, dtype = float)] + [np.power(base, power+1) for power in range(len(xs))]).T, y
    

def main():
    # Generate some dummy data
    linear, square = dummyLinear(c = 3.0), dummyTwo()
    #plt.scatter(np.linspace(0, 1, linear.shape[0]), linear)
    #plt.scatter(np.linspace(0, 1, square.shape[0]), square)

    x, y = dummyX()

    lin_model = sm.GLS(y, x)
    res = lin_model.fit()
    print(res.summary())
    print(res.params)

def calculate_survCorr(crisprDepsLoc: Optional[str] = None, hugoLoc: Optional[str] = None, cellInfoLoc: Optional[str] = None,
         gdsc1Loc: Optional[str] = None, gdsc2Loc: Optional[str] = None,
         logFile: Logger = Logger(os.path.join("Data", "Results", "GDSC-SC-calculation-output.txt")),
         dMode: bool = DEBUG_MODE):
    # First, update logFile path if in debug mode
    if(dMode):
        logFile.directory = logFile.directory[:-4]+"-DEBUG"+logFile.directory[-4:]
    
    # Clear LogFile and add initial line
    logFile.clear()
    logFile.add("GDSC Function starts")
    # Compile dictionary of relevant file locations
    fileLocs = {}
    for i, name in enumerate(["crispr", "hugo", "cellinfo", "gdsc1", "gdsc2"]):
        alternative = [crisprDepsLoc, hugoLoc, cellInfoLoc, gdsc1Loc, gdsc2Loc][i]
        if(alternative is not None):
            fileLocs[name] = alternative
        else:
            fileLocs[name] = [DEFAULT_CRISPR_FILE, DEFAULT_HUGO_FILE, DEFAULT_CELL_INFO_FILE, DEFAULT_DRUG1_FILE, DEFAULT_DRUG2_FILE][i]

    # Get number of CPU's to use for multiprocessing
    cpu_count = max(1, mp.cpu_count()-2)

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
    logFile.add(f'Updating {len(old_names)} archaic gene names in dependency data')
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

    for d in dList:
        # Get all available cell lines
        cs = [df[df["DRUG_NAME"]==d] for df in [drug2, drug1]]
        
        # Go through all genes from CRISPR dependencies DataFrame
        genes = crisprDeps.columns
        # If in Debug mode, only do this for the first 20 genes
        if(dMode):
            genes = genes[:20]
        for i, gn in enumerate(genes):
            
            # get dependencies (deps) for all available cell lines, as well as a list of cell lines which
            # were found within the deps DataFrame

            deps = [crisprDeps[crisprDeps.index.isin(cldf["ModelID"].values)][gn].reset_index() for cldf in cs]
            #print(deps)
            dep_names = []
            for cldf in deps:
                if(len(cldf)>0):
                    dep_names.append(list(cldf["ModelID"]))
                else:
                    dep_names.append(None)
            #dep_names = [list(cldf["ModelID"]) if len(cldf) > 0 else None for cldf in cs ]

            # Get DataFrames of whatever is being used for establishing correlations (i.e. pKi, IC50 or eMax values)
            responses = []
            for gdscdf, cldf, cldf_names in zip(cs, deps, dep_names):
                if(len(cldf)<1):
                    responses.append(None)
                    continue
                newResponse = gdscdf[(gdscdf["DRUG_NAME"] == d) & 
                                (gdscdf["ModelID"].isin(cldf_names))].drop_duplicates \
                            (subset=["ModelID"], keep="first")[["LN_IC50","ModelID"]]
                responses.append(cldf.merge(newResponse))
            
            # Get Pearson Correlations
            prs, pps = [], []

            for response in responses:
                if(response is None):
                    prs.append(None)
                    pps.append(None)
                    continue
                
                # Get the two different responses - which is the Cell Line dependencies and the "LN_IC50" (eMax, IC50 etc.) data
                x = np.array(response[gn])
                y = response["LN_IC50"]
                # If dealing with IC50 data, multiply is by -1 to transform it from LN(IC50) to pIC50 (pIC50 = -LN(IC50))
                if("IC50" in "LN_IC50".upper()):
                    y *= -1
                # If dealing with eMax data, take the natural log and multiply by -1 to obtain peMax (-Ln(eMax)) from eMax values
                elif("emax" in "LN_IC50".lower()):
                    y = -1 * np.log(y)
                
                plt.scatter(x, y)
                plt.title(f"{d}-{gn} responses")
                plt.xlabel("Cell Line CRISPR Dependency")
                plt.ylabel("pIC50")
                plt.show()
                #pr, pp = pearsonr(x, y)
                prs.append(pr)
                pps.append(pp)
            
            # Select which pearson correlation to use
            validCorr = False
            # Select the highest-priority dataset correlation
            for pr in prs:
                if(pr is not None):
                    #result.at[gn, d] = pr
                    validCorr = True
                    break
        if(DEBUG_MODE):
            return

if(__name__=="__main__"):
    main()