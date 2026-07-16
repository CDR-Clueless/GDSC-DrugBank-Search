#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 15 Jul 2026

@author: jds40
"""

import os
from io import StringIO
from copy import deepcopy
from typing import Optional
from collections.abc import Callable
import multiprocessing as mp

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy.stats import linregress
import json

from drug_search import get_data, update_hgnc
from logger import Logger

CLEANED_DATA_DIR: str = os.path.join("Data", "Laurence-Data")
DEFAULT_CRISPR_FILE: str = os.path.join(CLEANED_DATA_DIR,"CRISPRGeneDependency.csv")
DEFAULT_HUGO_FILE: str = os.path.join(CLEANED_DATA_DIR, "hgnc_complete_set.tsv")
DEFAULT_CELL_INFO_FILE: str = os.path.join(CLEANED_DATA_DIR, "Model.csv")
DEFAULT_DRUG1_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC1_drug_results_target_cleaned7.tsv')
DEFAULT_DRUG2_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC2_drug_results_target_cleaned7.tsv')
DEFAULT_DRUG_COMB_FILE: str = os.path.join("Data", "Raw Data", "GDSCC")
DEFAULT_MODEL_STATS_OUTPUT: str = os.path.join("Data", "Results", "GLM Model Results-High Variance")

DEBUG_MODE: bool = False
if(os.path.exists(os.path.join("Local", "localVars.json"))):
    with open(os.path.join("Local", "localVars.json"), "r") as f:
        DEBUG_MODE = bool(json.load(f)["DEBUG_MODE"])

def main():
    logFile: Logger = Logger()
    logFile.clear()
    logFile.add("Preparing to train GLMs to predict drug response")
    #sc = import_SC()
    ess = import_essentiality()
    # Get gene essentiality variances for each gene and then sort them
    # NOTE: np.argsort is used instead of the regular sorted(zip()) thing because np.var returns np.float64 data types which don't seem sortable by sorted() (that or the list is too big for sorted())
    essVariancesdict = {gene: np.var(ess[gene].values) for gene in ess.columns}
    ind = np.argsort(np.array(list(essVariancesdict.values())))
    essVariances = list(zip(np.array(list(essVariancesdict.keys()))[ind], np.array(list(essVariancesdict.values()))[ind]))
    # Do the same thing for Survivability Correlation
    #scVariancesDict = {gene: np.var(sc.loc[gene].values) for gene in sc.index}
    #ind = np.argsort(np.array(list(scVariancesDict.values())))
    #scVariances = list(zip(np.array(list(scVariancesDict.keys()))[ind], np.array(list(scVariancesDict.values()))[ind]))
    # Get top 100 most variable genes for each of the above lists in which most (>85%) of rows have valid values for it
    highVarianceEss = []
    for gene in [str(t[0]) for t in essVariances[::-1]]:
        prop = len(ess[gene][~np.isnan(ess[gene])]) / len(ess[gene])
        if(prop<0.85):
            #print(f"Discounting gene {gene} as only {prop*100}% of Cell Lines have a valid value")
            continue
        highVarianceEss.append(gene)
        if(len(highVarianceEss)>100):
            break

    #highVarianceSC = [str(t[0]) for t in scVariances[-100:]]
    # If in debug mode, reduce this to 10
    if(DEBUG_MODE):
        highVarianceEss = highVarianceEss[-10:]
        #highVarianceSC = highVarianceSC[-10:]
    # Get a refined essentiality dataframe for relevant genes for each cell line
    relEss = ess[highVarianceEss]
    # Drop Cell Lines with NaN values
    relEss = relEss.dropna(axis = 0)

    # Load in DataFrame to link Essentiality and GDSC1/2 data
    linker = import_linker()

    # Load in GDSC2 data
    gdsc2 = import_gdsc()

    # Ensure there's an output directory for the model results output
    if(not os.path.exists(DEFAULT_MODEL_STATS_OUTPUT)):
        os.mkdir(DEFAULT_MODEL_STATS_OUTPUT)

    # Train predictors for each different drug
    logFile.add("Training GLMs")
    perc, drugLen = 0.1, len(gdsc2.index.unique())
    for i, drug in enumerate(gdsc2.index.unique()):
        relGD = gdsc2.loc[gdsc2.index == drug]
        scores = {relGD["ModelID"].values[i]: relGD["pIC50"].values[i] for i in range(len(relGD))}
        predictorFrame = deepcopy(relEss)
        predictorFrame["pIC50"] = predictorFrame.index.map(scores)
        predictorFrame.dropna(axis = 0, inplace=True)

        results = train_gls(predictorFrame, "pIC50").summary()

        stats = pd.read_html(StringIO(results.tables[0].as_html()))[0]
        coefficients = pd.read_html(StringIO(results.tables[1].as_html()), header=0, index_col=0)[0]

        stats.to_csv(os.path.join(DEFAULT_MODEL_STATS_OUTPUT, f"{drug}-stats.tsv"), sep = "\t", lineterminator="\n", index = False, header = False)
        coefficients.to_csv(os.path.join(DEFAULT_MODEL_STATS_OUTPUT, f"{drug}-coefficients.tsv"), sep = "\t", lineterminator="\n", index = True)
        if(i/drugLen>perc):
            logFile.add(f"{perc*100}% of drugs trained on")
            perc += 0.1
    return

def import_essentiality() -> pd.DataFrame:
    ess = pd.read_csv(os.path.join("Data", "Laurence-Data", "CRISPRGeneDependency.csv"), sep = ",")
    ess.columns = [col.split(" ")[0].strip() for col in ess.columns]
    ess["Cell Line Name"] = ess["Unnamed:"]
    del ess["Unnamed:"]
    ess.set_index("Cell Line Name", inplace = True)
    toUpdate = pd.DataFrame(data = ess.columns, columns = ["Gene"])
    toUpdate, success = update_hgnc(toUpdate, column = "Gene", report_completion=True)
    #print(f"{success} genes updated successfully")
    ess.columns = toUpdate["Gene"].values
    return ess

def import_SC() -> pd.DataFrame:
    sc = pd.read_csv(os.path.join("Data", "Results", "Survivability-Correlations", "pIC50-AllDrugsByAllGenes.tsv"), sep = "\t")
    sc["Gene"] = sc["symbol"]
    del sc["symbol"]
    sc.set_index("Gene", inplace = True)
    return sc

def import_linker() -> pd.DataFrame:
    # Load in cell line information - useful for linking cell line names/IDs/etc. from other DataFrames to other information from other DataFrames
    clInfo = pd.read_csv(DEFAULT_CELL_INFO_FILE, low_memory=False).fillna('')
    clInfo['OncotreeLineage'] = [x.upper() for x in clInfo['OncotreeLineage']]
    clInfo["OncotreePrimaryDisease"] = clInfo["OncotreePrimaryDisease"].str.replace(' ','_')
    return clInfo

def import_gdsc(version: int = 2) -> pd.DataFrame:
    if(version==1):
        df = pd.read_csv(DEFAULT_DRUG1_FILE, sep = "\t")
    else:
        df = pd.read_csv(DEFAULT_DRUG2_FILE, sep = "\t")
    df["DRUG_NAME"] = df["DRUG_NAME"].apply(lambda x:x.upper().strip())
    df.set_index("DRUG_NAME", inplace=True)
    df["pIC50"] = np.multiply(df["LN_IC50"], -1)
    return df

def train_gls(df: pd.DataFrame, responseColumn: str, components: int = 1) -> sm.GLS:

    # First, separate the input and output frames
    y = pd.Series(df[responseColumn].values, name = "pIC50")
    del df[responseColumn]

    # Next, turn the remaining data into a 2D matrix
    x = df.to_numpy()

    # Next, add in columns depending on desired component count
    base = x
    for power in range(2, components+1):
        x = np.hstack(x, np.power(base, power))

    # To finish data preprocessing, add in a column of 1's to represent the intercept
    x = sm.add_constant(x)

    # Turn x back into a DataFrame for proper labels
    columns = ["Constant"] + list(df.columns)
    for i in range(2, components+1):
        columns += [f"{col}^{i}" for col in df.columns]
    
    x_frame = pd.DataFrame(data = x, columns = columns)

    # Now create and train the model
    model = sm.GLS(y, x_frame)
    res = model.fit()
    return res
    

# Load in GDSC-Essentiality Data
def load_gdsc_ess(desiredDrug: str, desiredGene: str,
                  gdsc1: Optional[str] = None, gdsc2: Optional[str] = None):
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

    # Load in GDSC data
    if(gdsc1 is None):
        drug1 = pd.read_table(fileLocs["gdsc1"], low_memory=False).fillna('')
    else:
        drug1 = gdsc1
    drug1["DRUG_NAME"] = drug1["DRUG_NAME"].apply(lambda x:x.upper())
    
    if(gdsc2 is None):
        drug2 = pd.read_table(fileLocs["gdsc2"], low_memory=False).fillna('')
    else:
        drug2 = gdsc2
    drug2["DRUG_NAME"] = drug2["DRUG_NAME"].apply(lambda x:x.upper())

    # Initilise drug by gene data
    dList = sorted(set(drug1["DRUG_NAME"]) | set(drug2["DRUG_NAME"]))
    if(desiredDrug not in dList):
        print(f"Desired drug '{desiredDrug}' not found in GDSC1 nor GDSC2")
        return None, None

    drug1.set_index("DRUG_NAME", inplace = True)
    drug2.set_index("DRUG_NAME", inplace = True)

    # Update GDSC1/2 databases usin

    if(desiredGene not in drug1.columns and desiredGene not in drug2.columns):
        print(f"Desired gene '{desiredGene}' not found in GDSC1 nor GDSC2")
        return None, None
    
    drug1["pIC50"] = np.multiply(-1, drug1["LN_IC50"])
    drug2["pIC50"] = np.multiply(-1, drug2["LN_IC50"])

    out = {}

    # Get GDSC1/2 data
    for gdscv in ["GDSC1", "GDSC2"]:
        # Get relevant dataset
        rel = {"GDSC1": drug1, "GDSC2": drug2}[gdscv]
    
        # Get all available cell lines
        cs = rel[rel.index==desiredDrug]
                        
        # get dependencies (deps) for all available cell lines, as well as a list of cell lines which
        # were found within the deps DataFrame
        deps = crisprDeps[crisprDeps.index.isin(cs["ModelID"].values)][desiredGene].reset_index()
        if(len(deps)>0):
            dep_names = deps["ModelID"]
        else:
            dep_names = None

        # Get DataFrames of whatever is being used for establishing correlations (i.e. pKi, IC50 or eMax values)
        response = cs[(cs.index == desiredDrug) & 
                    (cs["ModelID"].isin(dep_names))].drop_duplicates \
                    (subset=["ModelID"], keep="first")[["pIC50","ModelID"]]
        mID, ess = deps["ModelID"].values, deps[desiredGene].values
        response["Essentiality"] = response["ModelID"].map({mID[i]: ess[i] for i in range(len(deps))})

        # Make DataFrame to concatenate onto GDSC1/2
        if(response is None):
            out[gdscv] = None
            continue

        c1, c2, c3 = response["ModelID"].values, response["Essentiality"].values, response["pIC50"].values
        out[gdscv] = pd.DataFrame(data = list(zip(c1, c2, c3)), columns = ["ModelID", "Essentiality", "pIC50"])
    
    return out["GDSC1"], out["GDSC2"]


if(__name__=="__main__"):
    main()