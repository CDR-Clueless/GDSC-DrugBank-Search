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

CLEANED_DATA_DIR: str = os.path.join("Data", "Laurence-Data")
DEFAULT_CRISPR_FILE: str = os.path.join(CLEANED_DATA_DIR,"CRISPRGeneDependency.csv")
DEFAULT_HUGO_FILE: str = os.path.join(CLEANED_DATA_DIR, "hgnc_complete_set.tsv")
DEFAULT_CELL_INFO_FILE: str = os.path.join(CLEANED_DATA_DIR, "Model.csv")
DEFAULT_DRUG1_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC1_drug_results_target_cleaned7.tsv')
DEFAULT_DRUG2_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC2_drug_results_target_cleaned7.tsv')
DEFAULT_OUTPUT_DIR: str = os.path.join("Data", "Results")

def main(crisprDepsLoc: Optional[str] = None, hugoLoc: Optional[str] = None, cellInfoLoc: Optional[str] = None,
         gdsc1Loc: Optional[str] = None, gdsc2Loc: Optional[str] = None):
    
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

    # Load in GDSC data
    drug1 = pd.read_table(DEFAULT_DRUG1_FILE, low_memory=False).fillna('')
    drug1["DRUG_NAME"] = drug1["DRUG_NAME"].apply(lambda x:x.upper())
    
    drug2 = pd.read_table(DEFAULT_DRUG2_FILE, low_memory=False).fillna('')
    drug2["DRUG_NAME"] = drug2["DRUG_NAME"].apply(lambda x:x.upper())

    # Initilise drug by gene data
    dList = sorted(set(drug1["DRUG_NAME"]) | set(drug2["DRUG_NAME"]))
    
    allbyall = pd.DataFrame(columns=["symbol"]+dList)
    allbyall["symbol"] = list(crisprDeps.columns)
    allbyall = allbyall.fillna(0.0)
    allbyall.set_index("symbol", inplace=True)

    # Set up directory to store temporary calculations from parallel functions
    if(os.path.exists(os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap_store"))==False):
        os.mkdir(os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap_store"))

    # Record time before parallel running
    t_base = time.time()

    # Break the list of drugs/compounds into a smaller lists which are passed to a parallel function to calculate them
    batch_dlist = split_list(dList,cpu_count)  
    nested_dfs = mp.Pool(cpu_count).starmap_async(chunkDrugGeneFormatted,
            [(i,batch_dlist[i],crisprDeps,[drug2,drug1])
             for i in range(cpu_count)]).get()
    
    print(f'All by All took {((time.time())-t_base)/60.0:.4} min')
    
    allbyall = pd.concat(nested_dfs,axis=1)   
    
    print('Writing Drugs x Genes file)')
    allbyall.to_csv(os.path.join(DEFAULT_OUTPUT_DIR, 'AllDrugsByAllGenes.tsv'), sep='\t', index=True, header=True)
    print('Writing Genes x Drugs file)')
    allbyall = allbyall.T
    allbyall.index.names = ["drug"]
    allbyall.to_csv(os.path.join(DEFAULT_OUTPUT_DIR, 'AllGenesByAllDrugs.tsv'), sep='\t', index=True, header=True)
    
    # Delete the temporary data store
    for filename in os.listdir(os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap_store")):
        os.remove(os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap_store", filename))
    os.rmdir(os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap_store"))
    
    return

def chunkDrugGeneFormatted(it: int, il: set, CRISPRdeps: pd.DataFrame, drugFrames: list[pd.DataFrame],
                           priorityList: bool = True):
    """Altered version of ChunkDrugGene to calculate survival correlations for GDSC1/2 and GDSCC

    Args:
        it (int): (Simplified) ID for the thread working on this chunk
        il (set): List of indexes from the DrugFrames (i.e. list of drugs) to calculate SC's for
        deps (pd.DataFrame): DataFrame of CRISPR gene dependency data (DepMap)
        drugFrames (list[pd.DataFrame]): List of DataFrames with IC50/pKi data (i.e. GDSC(/C) data)
        priorityList (bool): Whether the drugFrames list is ordered in terms of importance
            ([most important, mid importance, least importance]) - if True, the highest-importance non-NaN correlation score is used. If False, the highest non-NaN correlation score is used
    """
    result = pd.DataFrame(columns=['symbol']+il)
    result.symbol = list(CRISPRdeps.columns)
    result = result.fillna(np.nan)
    result.set_index("symbol", inplace=True)

    # loop through all indexes, i.e. drugs/compounds, calculating r for all genes
    for d in tqdm(il, desc=f"Thread {it} progress"):
        # Load the calculation for this data if it has already been calculated
        starfiledir = os.path.join(DEFAULT_OUTPUT_DIR, "temp_starmap_store", f"starmapcorrelations-{d}")
        if(os.path.exists(starfiledir)):
           # Get finished results
           with open(starfiledir, "r") as f:
               dresult = str(f.read()).split("\n")
           # Temporary check to format results if the drug name is at the top
           if(dresult[0][0] not in ["0","-"]):
               dresult.pop(0)
           # Turn dresult from list of strings into list of floats
           dresult = [float(dresult[i]) for i in range(len(dresult))]
           result[:, d] = deepcopy(dresult)
           print(f"Thread {it} found and loaded correlations for {d}", flush = True)
           continue

        # If there is no file, calculate the correlations for this drug
        print(f'Thread {it} Calculating correlations for {d}',flush=True)

        # Get all available cell lines
        cs = [df[df["DRUG_NAME"]==d] for df in drugFrames]
        
        # Go through all genes from CRISPR dependencies DataFrame
        for i, gn in enumerate(CRISPRdeps.columns):
            
            # get dependencies (deps) for all available cell lines, as well as a list of cell lines which
            # were found within the deps DataFrame

            deps = [CRISPRdeps[CRISPRdeps.index.isin(cldf["ModelID"].values)][gn].reset_index() for cldf in cs]
            dep_names = []
            for cldf in deps:
                if(len(cldf)>0):
                    dep_names.append(list(cldf["ModelID"]))
                else:
                    dep_names.append(None)
            #dep_names = [list(cldf["ModelID"]) if len(cldf) > 0 else None for cldf in cs ]

            # Get pKi DataFrames
            pKis = []
            for gdscdf, cldf, cldf_names in zip(cs, deps, dep_names):
                if(len(cldf)<1):
                    pKis.append(None)
                    continue
                pKi = gdscdf[(gdscdf["DRUG_NAME"] == d) & 
                             (gdscdf["ModelID"].isin(cldf_names))].drop_duplicates \
                            (subset=["ModelID"], keep='first')[['pKi','ModelID']]
                pKis.append(cldf.merge(pKi))
            
            # Get Pearson Correlations
            prs, pps = [], []

            for pKi in pKis:
                if(pKi is None):
                    prs.append(None)
                    pps.append(None)
                    continue
                
                x = np.array(pKi[gn])
                y = pKi["pKi"]

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
            f.write("\n".join(result[d].values))
        #result[d].to_csv(starfiledir, sep = "\t", index = False, lineterminator="\n")
            
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
    n = min(parts, max(len(l),1))
    k, m = divmod(len(l), n)
    return [l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

if(__name__=="__main__"):
    main()