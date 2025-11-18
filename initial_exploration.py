#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2025

@author: jds40
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from copy import deepcopy
import json
from tqdm import tqdm
import ast

from typing import Optional

from data_handler import DataHandler
from searcher import Searcher
from drug_gene_correlation_histograms import CorrelationPlotter

CLEANED_DATA_DIR: str = os.path.join("Data", "Laurence-Data")

DEFAULT_HUGO_FILE: str = os.path.join(CLEANED_DATA_DIR, "hgnc_complete_set.tsv")
DEFAULT_ALL_BY_ALL_FILE: str = os.path.join(CLEANED_DATA_DIR, "AllDrugsByAllGenes.tsv")
DEFAULT_STRING_INFO_FILE: str = os.path.join(CLEANED_DATA_DIR, "9606.protein.info.v12.0.txt")
DEFAULT_STRING_LINK_FILE: str = os.path.join(CLEANED_DATA_DIR, "9606.protein.links.v12.0.ssv")

def main():
    ## Import relevant datasets and amend them
    # HGNC
    hgnc = pd.read_table(DEFAULT_HUGO_FILE, low_memory=False).fillna('')
    hgnc = hgnc[['symbol', 'ensembl_gene_id',
                'prev_symbol', 'location', 'location_sortable']]
    hgnc.set_index('symbol', inplace=True)

    # STRING proteins
    stinfo =  pd.read_table(DEFAULT_STRING_INFO_FILE,
                            usecols=['#string_protein_id','preferred_name']).fillna('')
    stinfo.rename(columns = {'#string_protein_id':'ID','preferred_name':'symbol'},inplace=True)
    stinfo = update_hgnc(stinfo, hgnc)

    stdict = dict(zip(stinfo.ID,stinfo.symbol))

    # STRING network
    stlink = pd.read_csv(DEFAULT_STRING_LINK_FILE,sep=' ')
    stlink.combined_score = stlink.combined_score.astype(int)
    # Change proteins in stlink from ID's to HGNC-checked names
    stlink.protein1 = stlink.protein1.apply(lambda x: stdict[x])
    stlink.protein2 = stlink.protein2.apply(lambda x: stdict[x])

    # DrugxGene survivability scores
    allbyall = pd.read_csv(DEFAULT_ALL_BY_ALL_FILE, sep = "\t")

    # Refine STRING links to those with a combined score >0.8
    stlink[stlink.combined_score.gt(800)]
    
    #plotter = CorrelationPlotter()
    #plotter.plot_all()
    return

def update_hgnc(df: pd.DataFrame, hgncdata: pd.DataFrame) -> pd.DataFrame:
    """
    
    Normalise archaic names using HGNC standard

    Args:
        df (pd.DataFrame): DataFrame with names to replace using HGNC
        hgncdata (pd.DataFrame): HGNC data

    Returns:
        pd.DataFrame: Version of 'df' with updated gene names
    """
    bad_names = set(df.symbol) & (set(df.symbol) ^ set(hgncdata.index))

    for g in bad_names:
        g2 = hgncdata[hgncdata['prev_symbol'].str.contains(g)].reset_index()['symbol']
        if len(g2) == 0 or (g2[0] not in hgncdata.index):
            print(f'STRING Gene name {g} not found in HUGO - ignoring it')
            continue
        else:
            print(f'STRING old gene name {g} replaced by new name {g2[0]}')
            df.replace(g,g2[0], inplace=True)
    return df

if __name__ == "__main__":
    main()
