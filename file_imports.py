#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 5 Nov 2026

@author: jds40
"""

import os
from typing import Optional

import pandas as pd

from logger import Logger

CLEANED_DATA_DIR: str = os.path.join("Data", "Laurence-Data")
DEFAULT_HUGO_FILE: str = os.path.join("Data", "Laurence-Data", "hgnc_complete_set.tsv")
DEFAULT_ALLBYALL: str = os.path.join("Data", "Results", "Survivability-Correlations", "GDSC", "pIC50-pearson-AllDrugsByAllGenes.tsv")
DEFAULT_CRISPR_FILE: str = os.path.join(CLEANED_DATA_DIR,"CRISPRGeneDependency.csv")

from drug_search import update_hgnc

def main():
    return

def import_allbyall(loc: str = DEFAULT_ALLBYALL, hugoLoc: str = DEFAULT_HUGO_FILE) -> pd.DataFrame:
    """Import Drug by Gene data, making sure gene names are updated with HUGO gene standardisation

    Args:
        loc (str, optional): _description_. Defaults to DEFAULT_ALLBYALL.
        hugoLoc (str, optional): _description_. Defaults to DEFAULT_HUGO_FILE.

    Returns:
        pd.DataFrame: _description_
    """
    # Import hugo file
    hgnc = pd.read_table(hugoLoc, low_memory=False).fillna('')
    hgnc = hgnc[['symbol', 'ensembl_gene_id',
                'prev_symbol', 'location', 'location_sortable']]
    hgnc.set_index('symbol', inplace=True)

    # Import Drug by Gene data and update with HUGO data
    dxg = pd.read_csv(loc, sep = "\t")
    # Ensure 'symbol' column is the genes
    if("symbol" not in dxg.columns):
        if("unnamed" in dxg.columns[0].lower()):
            dxg.rename(columns = {dxg.columns[0]: "symbol"}, inplace = True)
    dxg = update_hgnc(dxg, hgnc)
    dxg.set_index("symbol", inplace=True)
    return dxg

def import_crispr(crisprLoc: str = DEFAULT_CRISPR_FILE, hugoLoc: Optional[str] = DEFAULT_HUGO_FILE, 
                  logFile: Optional[Logger] = None, standardise: bool = True):
    # Get known CRISPR cell line-gene dependencies (row index = model ID/cell line ID, column = Gene)
    crisprDeps = pd.read_csv(crisprLoc).fillna(0.0)

    crisprDeps.rename(columns = {'Unnamed: 0':'ModelID'},inplace=True)
    crisprDeps.set_index('ModelID', inplace=True)
    
    # edit header names to remove spaces etc.
    gg = dict(zip(list(crisprDeps.columns), [i.strip().split()[0]
              for i in list(crisprDeps.columns)]))
    crisprDeps.rename(columns=gg, inplace=True)
    if(not standardise):
        return crisprDeps

    # Get HUGO standardised gene name information
    hgnc = pd.read_table(hugoLoc, low_memory=False).fillna('')
    hgnc = hgnc[['symbol', 'ensembl_gene_id',
                 'prev_symbol', 'location', 'location_sortable']]
    hgnc.set_index('symbol', inplace=True)
    
    # Correct legacy gene names in crisprDeps using HUGO table
    old_names = set(crisprDeps.columns) & (set(hgnc.index) ^ set(crisprDeps.columns))
    if(logFile is not None):
        logFile.add(f'Updating {len(old_names)} archaic gene names in dependency data')
    for g_old in old_names:
        g_new = hgnc[hgnc['prev_symbol'].str.contains(g_old)].reset_index()['symbol']
        if len(g_new) == 0 or (g_new[0] not in hgnc.index):
            continue
        else:
            crisprDeps.rename(columns={g_old: g_new[0]}, inplace=True)
    return crisprDeps

if(__name__=="__main__"):
    main()
