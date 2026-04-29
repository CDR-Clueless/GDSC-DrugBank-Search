#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 14:58:00 2026

@author: jds40
@coauthor: lp217 (Wrote original code)
"""

import os
from copy import deepcopy
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from tqdm import tqdm

import multiprocessing as mp

from typing import Union, Tuple, Optional

from logger import Logger

CLEANED_DATA_DIR: str = os.path.join("Data", "Laurence-Data")
DEFAULT_CRISPR_FILE: str = os.path.join(CLEANED_DATA_DIR,"CRISPRGeneDependency.csv")
DEFAULT_HUGO_FILE: str = os.path.join(CLEANED_DATA_DIR, "hgnc_complete_set.tsv")
DEFAULT_CELL_INFO_FILE: str = os.path.join(CLEANED_DATA_DIR, "Model.csv")
DEFAULT_DRUG1_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC1_drug_results_target_cleaned7.tsv')
DEFAULT_DRUG2_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC2_drug_results_target_cleaned7.tsv')
DEFAULT_DRUG_COMB_FILE: str = os.path.join("Data", "Raw Data", "GDSCC")
DEFAULT_OUTPUT_DIR: str = os.path.join("Data", "Results")

def main():
    #drugData, files = load_gdscc(DEFAULT_DRUG_COMB_FILE, returnLoaded = True)
    #combos, lines = len(drugData["Combo Name"].unique()), len(drugData["Cell Line Name"].unique())
    #print(f"{combos} unique combinations along {lines} cell lines")
    #print(drugData)
    gdsc()
    #gdscc()
    return

def load_gdscc(folderLoc: str = DEFAULT_DRUG_COMB_FILE, returnLoaded: bool = False,
               cellLineTranslators: list = [DEFAULT_DRUG1_FILE, DEFAULT_DRUG2_FILE]):
    df = pd.DataFrame(data = None, columns = ["Combo Name", "Cell Line Name", "Left Drug eMax", "Right Drug eMax", "Combo eMax",
                                              ])
    loaded_files = []
    # Go through available files in the directory
    for filename in os.listdir(folderLoc):
        # Check if this file is a GDSCC data file
        if(("anchor" in filename.lower() or "matrix" in filename.lower()) and filename.lower()[-4:] in [".csv", ".tsv"]):
            # Get appropriate separator for loading the DataFrame
            sep = {"csv": ",", "tsv": "\t"}[filename.lower()[-3:]]
            # Load the data into a consistent, more standardised format
            newdf = pd.read_csv(os.path.join(folderLoc, filename), sep = sep, low_memory=False)
            if("anchor" in filename.lower()):
                # Refine to relevant columns
                newdf = newdf[["Anchor Name", "Library Name", "Cell Line name", "Library Emax", "Combo Emax", "Anchor Conc"]]
                # Capitalise drug names to improve how standard they are
                newdf["Anchor Name"] = newdf["Anchor Name"].apply(lambda x: x.upper().strip())
                newdf["Library Name"] = newdf["Library Name"].apply(lambda x: x.upper().strip())
                # Combine Anchor and Library names into Combo name which is standardised by alphabet so duplicates can be removed later
                rows = []
                for anchor in newdf["Anchor Name"].unique():
                    anchored = newdf.loc[newdf["Anchor Name"]==anchor]
                    for library in anchored["Library Name"].unique():
                        tests = anchored.loc[anchored["Library Name"]==library]
                        # For each Cell Line experiment, only keep the one with the highest anchor concentration
                        for cl in tests["Cell Line name"].unique():
                            rel = tests.loc[tests["Cell Line name"]==cl]
                            bestConc = max(rel["Anchor Conc"].values)
                            results = rel.loc[rel["Anchor Conc"]>=bestConc]
                            for i in range(len(results)):
                                anchor, library = results["Anchor Name"].values[i], results["Library Name"].values[i]
                                # Note: NaN values are used for the anchor eMax's as the way this experiment works is keeping (sort of?) constant Anchor concentrations
                                if(anchor>library):
                                    rows.append(deepcopy([f"{anchor}###{library}", results["Cell Line name"].values[i],
                                                        np.nan, results["Library Emax"].values[i],
                                                        results["Combo Emax"].values[i]]))
                                else:
                                    rows.append(deepcopy([f"{library}###{anchor}", results["Cell Line name"].values[i],
                                                        results["Library Emax"].values[i], np.nan,
                                                        results["Combo Emax"].values[i]]))
                # Add this data into the main DataFrame
                df = pd.concat([df, pd.DataFrame(data = rows, columns = ["Combo Name", "Cell Line Name",
                                                                        "Left Drug eMax", "Right Drug eMax",
                                                                        "Combo eMax"])], ignore_index=True)
                loaded_files.append(filename)
            elif("matrix" in filename.lower()):
                # Refine to relevant columns
                newdf = newdf[["lib1_name", "lib2_name", "CELL_LINE_NAME", "lib1_MaxE", "lib2_MaxE", "combo_MaxE"]]
                # Capitalise drug names to improve standardisation
                newdf["lib1_name"] = newdf["lib1_name"].apply(lambda x: x.upper().strip())
                newdf["lib2_name"] = newdf["lib2_name"].apply(lambda x: x.upper().strip())
                # Combine library names into combo name which is standardised by alphabet so duplicates can be removed later
                rows = []
                for i in range(len(newdf)):
                    l1, l2, cln, l1emax, l2emax, cemax = [newdf[c].values[i] for c in 
                                                          ["lib1_name", "lib2_name", "CELL_LINE_NAME",
                                                           "lib1_MaxE", "lib2_MaxE", "combo_MaxE"]]
                    if(l1>l2):
                        rows.append(deepcopy([f"{l1}###{l2}", cln, l1emax, l2emax, cemax]))
                    else:
                        rows.append(deepcopy([f"{l2}###{l1}", cln, l2emax, l1emax, cemax]))
                # Add this to the main DataFrame
                df = pd.concat([df, pd.DataFrame(data = rows, columns = ["Combo Name", "Cell Line Name",
                                                                        "Left Drug eMax", "Right Drug eMax",
                                                                        "Combo eMax"])], ignore_index=True)
                loaded_files.append(filename)
    ## Now, translate the Cell Line Names into 'ModelID' which is more standardised and crosses with the CRISPR data
    translatorDict = {}
    for translator in cellLineTranslators:
        filetype = translator.split(".")[-1].lower()
        if(filetype == "tsv"):
            tdf = pd.read_csv(translator, sep = "\t")
        elif(filetype == "csv"):
            tdf = pd.read_csv(translator, sep = ",")
        for i in range(len(tdf)):
            if(tdf["CELL_LINE_NAME"].values[i] not in translatorDict.keys()):
                translatorDict[tdf["CELL_LINE_NAME"].values[i]] = tdf["ModelID"].values[i]
    df["ModelID"] = df["Cell Line Name"].map(translatorDict)
    # Return either the output DataFrame or that plus the list of loaded files if requested
    if(returnLoaded):
        return df, loaded_files
    return df

def gdscc(responseColumn: str = "eMax",
          crisprDepsLoc: Optional[str] = None, hugoLoc: Optional[str] = None, cellInfoLoc: Optional[str] = None,
         gdsccLoc: Optional[str] = None, cpu_count: int = max(1, mp.cpu_count()-2),
         logFile: Logger = Logger(os.path.join("Data", "Results", "GDSCC-SC-calculation-output.txt"))):
    
    # Clear logFile
    logFile.clear()
    
    # Compile dictionary of relevant file locations
    fileLocs = {}
    for i, name in enumerate(["crispr", "hugo", "cellinfo", "gdscc"]):
        alternative = [crisprDepsLoc, hugoLoc, cellInfoLoc, gdsccLoc][i]
        if(alternative is not None):
            fileLocs[name] = alternative
        else:
            fileLocs[name] = [DEFAULT_CRISPR_FILE, DEFAULT_HUGO_FILE, DEFAULT_CELL_INFO_FILE, DEFAULT_DRUG_COMB_FILE][i]

    # Get known CRISPR cell line-gene dependencies (row index = model ID/cell line ID, column = Gene)
    crisprDeps = pd.read_csv(DEFAULT_CRISPR_FILE).fillna(0.0)

    crisprDeps.rename(columns = {'Unnamed: 0':'ModelID'},inplace=True)
    crisprDeps.set_index('ModelID', inplace=True)
    
    # Edit header names to remove spaces etc.
    gg = dict(zip(list(crisprDeps.columns), [i.strip().split()[0]
              for i in list(crisprDeps.columns)]))
    crisprDeps.rename(columns=gg, inplace=True)

    # Get HUGO standardised gene name information
    hgnc = pd.read_table(DEFAULT_HUGO_FILE, low_memory=False).fillna('')
    hgnc = hgnc[['symbol', 'ensembl_gene_id',
                 'prev_symbol', 'location', 'location_sortable']]
    hgnc.set_index('symbol', inplace=True)
    
    # Correct legacy gene names in crisprDeps using HUGO table
    old_names = set(crisprDeps.columns) & (set(hgnc.index) ^ set(crisprDeps.columns))
    print(f'Updating {len(old_names)} archaic gene names in dependency data')
    for g_old in old_names:
        g_new = hgnc[hgnc['prev_symbol'].str.contains(g_old)].reset_index()['symbol']
        if len(g_new) == 0 or (g_new[0] not in hgnc.index):
            continue
        else:
            crisprDeps.rename(columns={g_old: g_new[0]}, inplace=True)
    
    # Load in cell line information - useful for linking cell line names/IDs/etc. from other DataFrames to other information from other DataFrames
    clInfo = pd.read_csv(DEFAULT_CELL_INFO_FILE, low_memory=False).fillna('')
    clInfo['OncotreeLineage'] = [x.upper() for x in clInfo['OncotreeLineage']]
    clInfo["OncotreePrimaryDisease"] = clInfo["OncotreePrimaryDisease"].str.replace(' ','_')

    # Separate out cancer types from cell line information as it is of particular note
    cancer_types = set(clInfo['OncotreeLineage'])

    # Load in GDSCC data
    drugData, files = load_gdscc(fileLocs["gdscc"], returnLoaded = True)

    # Get list of combination names
    comboList = sorted(set(drugData["Combo Name"]))
    
    ## Create DataFrames for each individual drug from drugData (drugData is used for calculating Combo SC values; the DataFrame below is used for calculating individual drugs)
    # Get 2 DataFrames: 1 for drugs on the left of the # and 1 for drugs on the right
    l, r = deepcopy(drugData), deepcopy(drugData)
    # Change to the appropriate name
    l["Name"] = [n.split("#")[0] for n in l["Combo Name"]]
    r["Name"] = [n.split("#")[-1] for n in r["Combo Name"]]
    # Get the appropriate response column value for each DataFrame
    l[responseColumn] = l[f"Left Drug {responseColumn}"]
    r[responseColumn] = r[f"Right Drug {responseColumn}"]
    # Delete extraneous/misleading columns
    for col in ["Combo Name", f"Left Drug {responseColumn}", f"Right Drug {responseColumn}", f"Combo {responseColumn}"]:
        del l[col]
        del r[col]
    # Merge DataFrames and remove duplicate experiments
    singleDrugData: pd.DataFrame = pd.concat([l, r], ignore_index=True)
    singleDrugData.reset_index()
    singleDrugData.drop_duplicates(inplace=True)
    singleDrugList = sorted(set(singleDrugData["Name"]))

    #allbyall = pd.DataFrame(columns=["symbol"]+comboList)
    #allbyall["symbol"] = list(crisprDeps.columns)
    ##allbyall = allbyall.fillna(0.0)
    #allbyall.set_index("symbol", inplace=True)

    # Modify combination drug data so it can be fed into same function as single drug data
    drugData.rename({"Combo Name": "Name", f"Combo {responseColumn}": responseColumn}, axis = 1, inplace = True)
    del drugData[f"Left Drug {responseColumn}"]
    del drugData[f"Right Drug {responseColumn}"]

    # Remove all NaN rows in the DataFrames
    lOrig, rOrig, drugDataOrig = l.shape, r.shape, drugData.shape
    l.dropna(inplace=True, subset = ["Name", responseColumn])
    r.dropna(inplace=True, subset = ["Name", responseColumn])
    drugData.dropna(inplace=True, subset = ["Name", responseColumn])
    lNew, rNew, drugDataNew = l.shape, r.shape, drugData.shape
    logFile.add(f"Original DataFrame shapes:\n  l: {lOrig}\n  r: {rOrig}\n  c: {drugDataOrig}\nDropNaN'd DataFrame shapes:\n  l: {lNew}\n  r: {rNew}\n  c: {drugDataNew}")

    # Record time before parallel running
    t_base, t_prev = time.time(), time.time()

    # Break the list of drugs/compounds into a smaller lists which are passed to a parallel function to calculate them
    batch_comboList = split_list(comboList, cpu_count)
    batch_singleList = split_list(singleDrugList, cpu_count)

    countCombo, countSingle = len(comboList), len(singleDrugList)
    logFile.add(f"Calculating Survivability Correlation values for {countCombo} Combinations made from {countSingle} individual Drugs")

    # Calcuate the allbyall DataFrames for the single and individual drug sets
    for df, batchList, drugType in zip([drugData, singleDrugData], [batch_comboList, batch_singleList], ["Combo", "Single"]):
        # Create directory for temporary parallel calculation storage
        tempDir = os.path.join(DEFAULT_OUTPUT_DIR, f"temp_starmap_store-GDSCC-{drugType}")
        if(not os.path.exists(tempDir)):
            os.mkdir(tempDir)
        # Get results
        nested_dfs = mp.Pool(cpu_count).starmap_async(chunkDrugGeneFormatted,
                [(i,batchList[i],crisprDeps,[df], "Name", "ModelID", responseColumn, True, tempDir)
                for i in range(cpu_count)]).get()
        
        logFile.add(f'All by All for {drugType} {responseColumn} took {((time.time())-t_prev)/60.0:.4} min')
        
        allbyall = pd.concat(nested_dfs,axis=1)

        logFile.add(f"Finished Creating allbyall file for {drugType}; {allbyall.shape[0]} rows by {allbyall.shape[1]} columns")
        
        dbgFile = os.path.join(DEFAULT_OUTPUT_DIR, f"GDSCC-{drugType}-{responseColumn}-AllDrugsByAllGenes.tsv")
        logFile.add(f"Writing Drugs x Genes file to {dbgFile}")
        allbyall.to_csv(dbgFile, sep='\t', index=True, header=True)
        
        allbyall = allbyall.T
        allbyall.index.names = ["drugCombination"]
        gbdFile = os.path.join(DEFAULT_OUTPUT_DIR, f"GDSCC-{drugType}-{responseColumn}-AllGenesByAllDrugs.tsv")
        logFile.add(f"Writing Genes x Drugs file to {gbdFile}")
        allbyall.to_csv(gbdFile, sep='\t', index=True, header=True)
        # Delete the temporary data storage
        for filename in os.listdir(tempDir):
            os.remove(os.path.join(tempDir, filename))
        os.rmdir(tempDir)
        t_prev = time.time()
    logFile.add("Finished calculating GDSCC correlations")

def gdsc(crisprDepsLoc: Optional[str] = None, hugoLoc: Optional[str] = None, cellInfoLoc: Optional[str] = None,
         gdsc1Loc: Optional[str] = None, gdsc2Loc: Optional[str] = None,
         logFile: Logger = Logger(os.path.join("Data", "Results", "GDSC-SC-calculation-output.txt"))):
    
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
    crisprDeps = pd.read_csv(DEFAULT_CRISPR_FILE).fillna(0.0)

    crisprDeps.rename(columns = {'Unnamed: 0':'ModelID'},inplace=True)
    crisprDeps.set_index('ModelID', inplace=True)
    
    # edit header names to remove spaces etc.
    gg = dict(zip(list(crisprDeps.columns), [i.strip().split()[0]
              for i in list(crisprDeps.columns)]))
    crisprDeps.rename(columns=gg, inplace=True)

    # Get HUGO standardised gene name information
    hgnc = pd.read_table(DEFAULT_HUGO_FILE, low_memory=False).fillna('')
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
    clInfo = pd.read_csv(DEFAULT_CELL_INFO_FILE, low_memory=False).fillna('')
    clInfo['OncotreeLineage'] = [x.upper() for x in clInfo['OncotreeLineage']]
    clInfo["OncotreePrimaryDisease"] = clInfo["OncotreePrimaryDisease"].str.replace(' ','_')

    # Separate out cancer types from cell line information as it is of particular note
    cancer_types = set(clInfo['OncotreeLineage'])

    # Load in GDSC data
    drug1 = pd.read_table(DEFAULT_DRUG1_FILE, low_memory=False).fillna('')
    drug1["DRUG_NAME"] = drug1["DRUG_NAME"].apply(lambda x:x.upper())
    
    drug2 = pd.read_table(DEFAULT_DRUG2_FILE, low_memory=False).fillna('')
    drug2["DRUG_NAME"] = drug2["DRUG_NAME"].apply(lambda x:x.upper())

    # Initilise drug by gene data
    dList = sorted(set(drug1["DRUG_NAME"]) | set(drug2["DRUG_NAME"]))
    
    #allbyall = pd.DataFrame(columns=["symbol"]+dList)
    #allbyall["symbol"] = list(crisprDeps.columns)
    #allbyall = allbyall.fillna(0.0)
    #allbyall.set_index("symbol", inplace=True)


    # Set up directory to store temporary calculations from parallel functions
    if(os.path.exists(os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap_store"))==False):
        os.mkdir(os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap_store"))

    # Record time before parallel running
    t_base, t_prev = time.time(), time.time()

    # Break the list of drugs/compounds into a smaller lists which are passed to a parallel function to calculate them
    batch_dlist = split_list(dList,cpu_count)
    logFile.add("Running GDSC Parallel code")
    for responseColumn in ["pKi", "LN_IC50"]:
        logFile.add(f"All by all running for {responseColumn}")
        nested_dfs = mp.Pool(cpu_count).starmap_async(chunkDrugGeneFormatted,
                [(i,batch_dlist[i],crisprDeps,[drug2,drug1],
                "DRUG_NAME", "ModelID", responseColumn, True, None, logFile)
                for i in range(cpu_count)]).get()
        
        logFile.add(f'{responseColumn} All by All took {((time.time())-t_prev)/60.0:.4} min')
        
        allbyall = pd.concat(nested_dfs,axis=1)   
        
        logFile.add('Writing Drugs x Genes file)')
        allbyall.to_csv(os.path.join(DEFAULT_OUTPUT_DIR, f"{responseColumn}-AllDrugsByAllGenes.tsv"), sep='\t', index=True, header=True)
        logFile.add('Writing Genes x Drugs file)')
        allbyall = allbyall.T
        allbyall.index.names = ["Drug"]
        allbyall.to_csv(os.path.join(DEFAULT_OUTPUT_DIR, f"{responseColumn}-AllGenesByAllDrugs.tsv"), sep='\t', index=True, header=True)
        # Delete the temporary data store
        for filename in os.listdir(os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap_store")):
            os.remove(os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap_store", filename))
        os.rmdir(os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap_store"))
        t_prev = time.time()
    
    logFile.add(f"GDSC Calculation finished. Total time taken {((time.time())-t_prev)/60.0:.4} minutes")
    
    return

def chunkDrugGeneFormatted(it: int, il: set, CRISPRdeps: pd.DataFrame, drugFrames: list[pd.DataFrame],
                           drugColumn: str = "DRUG_NAME", cellLineColumn: str = "ModelID", responseColumn: str = "pKi",
                           priorityList: bool = True, starfiledirBase: Optional[str] = None, logFile: Optional[Logger] = None):
    """Altered version of ChunkDrugGene to calculate survival correlations for GDSC1/2 and GDSCC

    Args:
        it (int): (Simplified) ID for the thread working on this chunk
        il (set): List of indexes from the DrugFrames (i.e. list of drugs) to calculate SC's for
        deps (pd.DataFrame): DataFrame of CRISPR gene dependency data (DepMap)
        drugFrames (list[pd.DataFrame]): List of DataFrames with IC50/pKi data (i.e. GDSC(/C) data)
        priorityList (bool): Whether the drugFrames list is ordered in terms of importance
            ([most important, mid importance, least importance]) - if True, the highest-importance non-NaN correlation score is used. If False, the highest non-NaN correlation score is used
    """
    
    if(logFile is not None):
        logFile.add(f"Thread {it} calculating correlation coefficient for {responseColumn}")

    result = pd.DataFrame(columns=['symbol']+il)
    result.symbol = list(CRISPRdeps.columns)
    result = result.fillna(np.nan)
    result.set_index("symbol", inplace=True)

    # loop through all indexes, i.e. drugs/compounds, calculating r for all genes
    for d in tqdm(il, desc=f"Thread {it} progress"):
        ## Load the calculation for this data if it has already been calculated
        if(starfiledirBase is None):
            starfiledir = os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap_store",
                f"starmapcorrelations-drugColumn_{drugColumn}-cellLineColumn_{cellLineColumn}-responseColumn_{responseColumn}-drug_{d}")
        else:
            starfiledir = os.path.join(starfiledirBase, f"starmapcorrelations-drugColumn_{drugColumn}-cellLineColumn_{cellLineColumn}-responseColumn_{responseColumn}-drug_{d}")
        if(os.path.exists(starfiledir)):
           # Get finished results
           with open(starfiledir, "r") as f:
               dresult = str(f.read()).split("\n")
           # First, ensure the results aren't empty - fill the results with np.nan values if not
           if(len(dresult)>1):
                # Temporary check to format results if the drug name is at the top
                if(dresult[0][0] not in ["0","-"]):
                    dresult.pop(0)
                # Turn dresult from list of strings into list of floats
                dresult = [dresult[i].replace("\"", "").replace("\'", "") for i in range(len(dresult))]
                dresult = [float(dresult[i]) if dresult[i] != "" else float("NaN") for i in range(len(dresult))]
                # Trim down to remove any spillover values caused by excess newlines in file writing
                dresult = dresult[:len(result[d])]
                # Copy results in
                result[d] = deepcopy(dresult)
                print(f"Thread {it} found and loaded correlations for {d}", flush = True)
                continue
           # If the results are blank, fill this with np.nan values
           else:
                result[d] = deepcopy([np.nan]*len(result[d]))

        ## If there is no file, calculate the correlations for this drug
        print(f'Thread {it} Calculating correlations for {d}',flush=True)

        # Get all available cell lines
        cs = [df[df[drugColumn]==d] for df in drugFrames]
        #print(cs)
        
        # Go through all genes from CRISPR dependencies DataFrame
        for i, gn in enumerate(CRISPRdeps.columns):
            
            # get dependencies (deps) for all available cell lines, as well as a list of cell lines which
            # were found within the deps DataFrame

            deps = [CRISPRdeps[CRISPRdeps.index.isin(cldf[cellLineColumn].values)][gn].reset_index() for cldf in cs]
            #print(deps)
            dep_names = []
            for cldf in deps:
                if(len(cldf)>0):
                    dep_names.append(list(cldf[cellLineColumn]))
                else:
                    dep_names.append(None)
            #dep_names = [list(cldf[cellLineColumn]) if len(cldf) > 0 else None for cldf in cs ]

            # Get DataFrames of whatever is being used for establishing correlations (i.e. pKi, IC50 or eMax values)
            responses = []
            for gdscdf, cldf, cldf_names in zip(cs, deps, dep_names):
                if(len(cldf)<1):
                    responses.append(None)
                    continue
                newResponse = gdscdf[(gdscdf[drugColumn] == d) & 
                             (gdscdf[cellLineColumn].isin(cldf_names))].drop_duplicates \
                            (subset=[cellLineColumn], keep="first")[[responseColumn,cellLineColumn]]
                responses.append(cldf.merge(newResponse))
            
            # Get Pearson Correlations
            prs, pps = [], []

            for response in responses:
                if(response is None):
                    prs.append(None)
                    pps.append(None)
                    continue
                
                x = np.array(response[gn])
                y = response[responseColumn]

                pr, pp = pearsonr(x, y)
                prs.append(pr)
                pps.append(pp)
            
            # Select which pearson correlation to use
            validCorr = False
            # If the source (GDSC) DataFrames are speicified in priority order, select the highest-priority dataset correlation
            if(priorityList):
                for pr in prs:
                    if(pr is not None):
                        result.at[gn, d] = pr
                        validCorr = True
                        break
            # If the DataFrames are not given in priority, choose the most non-zero correlation value
            else:
                result.at[gn, d] = max([pr for pr in prs if pr is not None], key = abs)
                validCorr = True
            # If we haven't found a correlation here (i.e. there was no valid Pearson correlation found), make the result NaN
            if(not validCorr):
                result.at[gn, d] = np.nan

        # Save result for this valud of 'd', in case the program is interrupted
        with open(starfiledir, "w") as f:
            f.write("\n".join([str(v) for v in result[d].values]))
            
    return(result)


def ChunkDrugGene(it: int, dl: set, deps: pd.DataFrame, drug1: pd.DataFrame, drug2: pd.DataFrame):
    """
    
    Function for calculating Drug-gene correlations for a chunk of drugs (necessary for parallel processing)

    Args:
        it (int): (Simplified) ID for the thread working on this chunk
        dl (set): Set of drugs in this chunk to be tested
        deps (pd.DataFrame): DataFrame of CRISPR gene dependency data (DepMap)
        drug1 (pd.DataFrame): DataFrame of GDSC1 data
        drug2 (pd.DataFrame): DataFrame of GDSC2 data
    """

    result = pd.DataFrame(columns=['symbol']+dl)
    result.symbol = list(deps.columns[1:])
    result = result.fillna(0.0)
    result.set_index('symbol', inplace=True)

    # loop through all drugs, calculating r for all genes
    for d in dl:
        # Load the calculation for this data if it has already been calculated
        starfiledir = os.path.join(CLEANED_DATA_DIR, "temp_starmap_store", f"starmapcorrelations-{d}")
        if(os.path.exists(starfiledir)):
           dresult = pd.read_csv(os.path.join(CLEANED_DATA_DIR, "temp_starmap_store", f""), sep = "\t", index = False, lineterminator="\n")
           result[:, d] = deepcopy(dresult)
           print(f"Thread {it} found and loaded correlations for {d}", flush = True)
           continue

        # If there is no file, calculate the correlations for this drug
        print(f'Thread {it} Calculating correlations for {d}',flush=True)

        # Get all available cell lines
        c1 = drug1[drug1.DRUG_NAME == d]['ModelID']
        c2 = drug2[drug2.DRUG_NAME == d]['ModelID']
        
        # Go through all genes from CRISPR dependencies DataFrame
        for i, gn in enumerate(deps.columns):
            
            # get dependencies (deps) for all available cell lines, as well as a list of cell lines which
            # were found within the deps DataFrame
            dep1 = deps[deps.index.isin(c1)][gn].reset_index()
            dep1_names = list(dep1.ModelID)
            dep2 = deps[deps.index.isin(c2)][gn].reset_index()
            dep2_names = list(dep2.ModelID)
            
            # get pKis
            if len(dep1) > 0:
                # Get first available pKi value for relevant drug with acceptable ModelID values
                pKi1 = drug1[(drug1.DRUG_NAME == d) & 
                             (drug1.ModelID.isin(dep1_names))].drop_duplicates \
                            (subset=["ModelID"], keep='first')[['pKi','ModelID']]
                # merge data on common ModelID
                # This means the merged frame will only include cell lines found in CRISPR dependency df
                # and should have both CRISPR-based gene dependency data and pKi data for each cell line
                dpdat1 = dep1.merge(pKi1)
                
            if len(dep2) > 0:
                pKi2 = drug2[(drug2.DRUG_NAME == d) & 
                             (drug2.ModelID.isin(dep2_names))].drop_duplicates \
                            (subset=["ModelID"], keep='first')[['pKi','ModelID']]
                dpdat2 = dep2.merge(pKi2)
                
            pr1,pp1,pr2,pp2 = 0,1,0,1

            if len(dep1) > 0:
                
                # Correlation coefficient between current gene expression (0 or 1 as it's binary k/o) and pKi value
                x = np.array(dpdat1[gn])
                y = dpdat1['pKi']
                                
                pr1,pp1 = pearsonr(x,y)

            if len(dep2) > 0:
                
                x = np.array(dpdat2[gn])
                # X = x[:,np.newaxis]
                y = dpdat2['pKi']
                
                pr2,pp2 = pearsonr(x,y)
                
            # select which value to use; pr1 or pr2
            result.at[gn,d] = max(pr1,pr2,key=abs)
            if (i % 1000) == 0:
                print(f'Thread {it} done {i} genes',flush=True)

        # Save result for this valud of 'd', in case the program is interrupted
        result[d].to_csv(starfiledir, sep = "\t", index = False, lineterminator="\n")
            
    return(result)

def split_list(l: list, parts: int) -> list:
    """Split a larger list into a given number of component lists (used here for more efficient multiprocessing batches)

    Args:
        l (list): A list
        parts (int): Number of component lists to break the larger list into

    Returns:
        list: _description_
    """
    # Number of parts to break this into; if we want to split it into more parts than there are, we'll need to add blank lists
    n = min(parts, max(len(l),1))
    k, m = divmod(len(l), n)
    output = [l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
    # Add extra empty lists if needed
    while(n<parts):
        output.append([])
        n += 1
    return output

if(__name__=="__main__"):
    main()