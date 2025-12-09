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
from scipy.optimize import curve_fit
import seaborn as sns
from copy import deepcopy
import json
from tqdm import tqdm
import ast
import igraph as ig
import diptest
import multiprocessing as mp

from typing import Optional

from data_handler import DataHandler
from searcher import Searcher
from drug_gene_correlation_histograms import CorrelationPlotter, curve_guess
from drug_search import update_hgnc, get_data

CLEANED_DATA_DIR: str = os.path.join("Data", "Laurence-Data")

DEFAULT_HUGO_FILE: str = os.path.join(CLEANED_DATA_DIR, "hgnc_complete_set.tsv")
DEFAULT_ALL_BY_ALL_FILE: str = os.path.join(CLEANED_DATA_DIR, "AllDrugsByAllGenes.tsv")
DEFAULT_STRING_INFO_FILE: str = os.path.join(CLEANED_DATA_DIR, "9606.protein.info.v12.0.txt")
DEFAULT_STRING_LINK_FILE: str = os.path.join(CLEANED_DATA_DIR, "9606.protein.links.v12.0.ssv")

#TARGET_DRUG: str = "965-D2"
#START_POINT_CUTOFF: float = 0.197

def initial_setup() -> None:
    """
    
    Do all necessary setup for scripts to properly run

    """
    # Ensure passwords json exists
    pdir = os.path.join("Local", "passwords.json")
    if(os.path.exists(pdir)==False):
        with open(pdir, "w") as f:
            json.dump({}, f)
    # Ensure passwords json has all necessary variables
    with open(pdir, "r") as f:
        pcont = json.load(f)
    if("core-count" not in pcont.keys()):
        pcont["core-count"] = "auto"
    with open(pdir, "w") as f:
        json.dump(pcont, f)
    return


def gaussian(x, A, mu, sigma):
    return A*np.exp(-np.divide(np.power(x-mu, 2),(2*np.power(sigma, 2))))

def bimodal(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2)

def main():
    # Perform initial setup
    initial_setup()

    #CorrelationPlotter().plot_all()
    #return
    

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

    # DataFrame for the gene/protein targets of all drugs
    drugTargets = pd.read_csv(os.path.join(CLEANED_DATA_DIR, "TargetRanking.tsv"), sep = "\t")
    drugTargets = update_hgnc(drugTargets, hgnc, "TARGET")

    # Dictionary of results for drug-gene survivability distributions
    with open(os.path.join("Data", "Results", "Drug-gene correlation frequency histograms", "stats.json"), "r") as f:
        drugGeneSurv = json.load(f)

    ## Prepare graphs
    # Refine STRING links to those with a combined score >0.8
    stlink[stlink.combined_score.gt(800)]
    
    print('Building human link graph ...')
    g_global = ig.Graph.TupleList(stlink.itertuples(index=False), directed=True, weights=False, edge_attrs="combined_score")
    print('Done !!')
    # Make path for temporary data storage if none exists
    tdpfp: str = os.path.join("Data", "Results", "drug_path_temp")
    if(os.path.exists(tdpfp)==False): 
        os.mkdir(tdpfp)
    for drug in tqdm(allbyall.columns, desc = "Calculating shortest drug paths"):
        # Check if data for this drug already exists
        if(os.path.exists(os.path.join(tdpfp, f"{drug}.json"))):
            print(f"Found drug path data already calculated for {drug}; skipping...")
            continue
        drugResults = {}
        # Get the appropriate threshold for 'starting point' genes
        survivability_cutoff = get_drug_threshold(drug, drugGeneSurv, np.array(allbyall[drug].values, dtype = float))
        # Get drug targets and save results output
        targets = drugTargets.loc[drugTargets["DRUG"]==drug]["TARGET"].values
        for target in targets:
            g = deepcopy(g_global)
            g.vs["survivability"] = [allbyall[drug].loc[gn] if gn in allbyall[drug].index else float("NaN") for gn in g.vs["name"]]
            startPoints = [n for s,n in zip(g.vs["survivability"], g.vs["name"]) if s > survivability_cutoff]
            shortest, shortestpath = np.inf, []
            for sp in startPoints:
                shortpath = g.get_shortest_paths(sp, to=target, weights=g.es["combined_score"], output="vpath")[0]
                if(shortest>len(shortpath)):
                    shortestpath = deepcopy(shortpath)
                    shortest = len(shortpath)
            # Get the nodes in this shortest path
            pathnodes = [g.vs[v] for v in shortestpath]
            drugResults[target] = { "path length": len(pathnodes)-1,
                                    "path": \
                                        {f"Path node {i}": {"name": pathnodes[i].attributes()["name"],
                                                "survivability": pathnodes[i].attributes()["survivability"]} \
                                                        for i in range(len(pathnodes))}}
        with open(os.path.join(tdpfp, f"{drug}.json"), "w") as f:
            json.dump(drugResults, f)
    
    ## Combine the per-drug jsons into a single, human-readable json and delete the originals
    # Combine per-drug jsons
    jsonContents = {}
    for filename in os.listdir(tdpfp):
        filedir = os.path.join(tdpfp, filename)
        drug = filedir.replace(".json","")
        with open(filedir, "r") as f:
            drugDict = json.load(f)
        jsonContents[drug] = deepcopy(drugDict)
    # Save results to json
    with open(os.path.join("Data", "Results", "drug paths.json"), "w") as f:
        json.dump(jsonContents, f, indent = 4)
    # Delete constituent parts
    for filename in os.listdir(tdpfp):
        filedir = os.path.join(tdpfp, filename)
        os.remove(filedir)
    os.rmdir(tdpfp)
    
    #USE HGNC ON DRUGBANK COMPARISON OUTPUT AND EXTEND SHORTEST PATHFINDING TO ALL TARGETS
    return

def get_drug_threshold(drug: str, drugSurvivabilityDict: dict, survivability_array: Optional[np.ndarray]) -> float:
    # Try to use the available data to get the appropriate cutoff
    try:
        rel = drugSurvivabilityDict[drug]
        # If the modality is unimodal or unclear, use 3 SDs above the mean as the threshold
        if(rel["modality details"]["modality"]!="bimodal"):
            return rel["modality details"]["mean"] + rel["standard deviations"]["3.0"]
        # If the modality is bimodal, use the mean of the higher curve survivability as the threshold
        else:
            return rel["curve parameters"]["mu1"]
    # If unsuccessful, try to just use 3 SDs above the mean
    except Exception as e:
        print(f"Failed to retrieve modality information for {drug}; defaulting to 3SDs above norm...")
        return float(np.mean(survivability_array)) + (float(np.std(survivability_array))*3.)

if __name__ == "__main__":
    main()
