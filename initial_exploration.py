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
import igraph as ig
import diptest

from typing import Optional

from data_handler import DataHandler
from searcher import Searcher
from drug_gene_correlation_histograms import CorrelationPlotter
from drug_search import update_hgnc, get_data

CLEANED_DATA_DIR: str = os.path.join("Data", "Laurence-Data")

DEFAULT_HUGO_FILE: str = os.path.join(CLEANED_DATA_DIR, "hgnc_complete_set.tsv")
DEFAULT_ALL_BY_ALL_FILE: str = os.path.join(CLEANED_DATA_DIR, "AllDrugsByAllGenes.tsv")
DEFAULT_STRING_INFO_FILE: str = os.path.join(CLEANED_DATA_DIR, "9606.protein.info.v12.0.txt")
DEFAULT_STRING_LINK_FILE: str = os.path.join(CLEANED_DATA_DIR, "9606.protein.links.v12.0.ssv")

TARGET_DRUG: str = "965-D2"
START_POINT_CUTOFF: float = 0.197

def main():
    
    """
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
    allbyall = update_hgnc(allbyall, hgnc)
    allbyall = allbyall.set_index("symbol")

    # Refine STRING links to those with a combined score >0.8
    stlink[stlink.combined_score.gt(800)]
    
    print('Building human link graph ...')
    g = ig.Graph.TupleList(stlink.itertuples(index=False), directed=True, weights=False, edge_attrs="combined_score")
    print('Done !!')
    g.vs["survivability"] = [allbyall[TARGET_DRUG].loc[gn] if gn in allbyall[TARGET_DRUG].index else float("NaN") for gn in g.vs["name"]]
    print(ig.summary(g))
    startPoints = [n for s,n in zip(g.vs["survivability"], g.vs["name"]) if s > START_POINT_CUTOFF]
    print(startPoints)
    shortest = np.inf
    for sp in startPoints:
        shortpath = g.get_shortest_paths(sp, to="MKNK1", weights=g.es["combined_score"], output="vpath")[0]
        shortest = min(shortest, len(shortpath)-1)
    
    #USE HGNC ON DRUGBANK COMPARISON OUTPUT AND EXTEND SHORTEST PATHFINDING TO ALL TARGETS
    """
    return

if __name__ == "__main__":
    main()
