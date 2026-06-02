#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created June 2026

@author: jds40
"""

import os
from copy import deepcopy
from typing import Optional
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import json
import igraph as ig

from tqdm import tqdm

from drug_search import update_hgnc
from logger import Logger
from survivability_correlation_calculation import split_list

DEFAULT_HUGO_FILE: str = os.path.join("Data", "Laurence-Data", "hgnc_complete_set.tsv")
DEFAULT_STRING_INFO_FILE: str = os.path.join("Data", "Laurence-Data", "9606.protein.info.v12.0.txt")
DEFAULT_STRING_LINK_FILE: str = os.path.join("Data", "Laurence-Data", "9606.protein.links.v12.0.ssv")
DEFAULT_ALLBYALL: str = os.path.join("Data", "Laurence-Data", "AllDrugsByAllGenes.tsv")


def main(saveDir: str = os.path.join("Data", "Results", "Target-Analysis"), logFile: Optional[Logger] = None,
         coreCount: int = max(mp.cpu_count() - 2, 1)):
    # Check if we're in debug mode
    if(os.path.exists(os.path.join("Local", "localVars.json"))):
        with open(os.path.join("Local", "localVars.json"), "r") as f:
            DEBUG_MODE: bool = json.load(f)["DEBUG_MODE"]
        # If in debug mode, go down to 2 processes to prevent issues
        if(DEBUG_MODE):
            coreCount = 2
    else:
        DEBUG_MODE: bool = False
    
    t_start = time.time()
    
    # Set up directories
    if(not os.path.exists(saveDir)):
        os.mkdir(saveDir)
    if(not os.path.exists(os.path.join(saveDir, "Gene-Paths"))):
        os.mkdir(os.path.join(saveDir, "Gene-Paths"))

    # Set up logger if custom directory not given
    if(logFile is None):
        logFile = Logger(os.path.join(saveDir, "Gene Paths Logfile.txt"))
    # Clear log file for this run
    logFile.clear()

    ## Prepare main STRING grpaph
    # Fetch HGNC/HUGO for updating gene names
    logFile.add("Importing HGNC file")
    hgnc = pd.read_table(DEFAULT_HUGO_FILE, low_memory=False).fillna('')
    hgnc = hgnc[['symbol', 'ensembl_gene_id',
                'prev_symbol', 'location', 'location_sortable']]
    hgnc.set_index('symbol', inplace=True)

    # Fetch list of genes from GDSC to check
    genes = pd.read_csv(DEFAULT_ALLBYALL, sep = "\t")
    genes = update_hgnc(genes, hgnc)
    genes = genes["symbol"].values
    # Reduce gene count to first 10 if in debug mode
    if(DEBUG_MODE):
        genes = genes[:10]

    # STRING proteins
    logFile.add("Importing STRING data")
    stinfo =  pd.read_table(DEFAULT_STRING_INFO_FILE,
                            usecols=['#string_protein_id','preferred_name']).fillna('')
    stinfo.rename(columns = {'#string_protein_id':'ID','preferred_name':'symbol'},inplace=True)
    stinfo = update_hgnc(stinfo, hgnc)

    # Link STRING ID's and HUGO-corrected Gene names
    stdict = dict(zip(stinfo.ID,stinfo.symbol))

    # STRING network
    stlink = pd.read_csv(DEFAULT_STRING_LINK_FILE,sep=' ')
    stlink.combined_score = stlink.combined_score.astype(int)
    # Change proteins in stlink from ID's to HGNC-checked names
    stlink.protein1 = stlink.protein1.apply(lambda x: stdict[x])
    stlink.protein2 = stlink.protein2.apply(lambda x: stdict[x])
    # Refine STRING links to those with a combined score >0.6
    stlink[stlink.combined_score.gt(600)]
    
    # Convert the STRING connectivity information to a graph with igraph
    logFile.add('Building human link graph ...')
    g_global = ig.Graph.TupleList(stlink.itertuples(index=False), directed=True, weights=False, edge_attrs="combined_score")
    logFile.add('Done !!')
    # Make sure the graph is sufficiently connected for our purposes
    if(not g_global.is_connected()):
        logFile.add("STRING network graph is not fully connected; try lowering score requirement or removing unneeded cliques")
        return
    # Set up temporary output directory for gene path calculation done in parallel
    tdpfp: str = os.path.join("Data", "Results", "Target-Analysis", "temporary store")
    if(os.path.exists(tdpfp)==False):
        os.mkdir(tdpfp)
    # Make/get record of genes already checked so they can be skipped
    # Dictionary is used rather than list for O(1) lookup times in later code
    genesChecked: dict = {}
    if(not os.path.exists(os.path.join(saveDir, "Gene-Paths"))):
        os.mkdir(os.path.join(saveDir, "Gene-Paths"))
    for filename in os.path.join(saveDir, "Gene-Paths"):
        genesChecked[filename.split("_")[0]] = True

    tocheck = [gene for gene in genes if gene not in genesChecked]

    batchList: list = split_list(tocheck, coreCount)

    logFile.add(f"Calculating shortest paths for {len(tocheck)} genes between {len(genes)} total genes")

    mp.Pool(coreCount).starmap_async(pathCheckWorker,
                    [(i, batchList[i], genes, os.path.join(saveDir, "Gene-Paths"), g_global, logFile)
                    for i in range(coreCount)]).get()
    
    timeTaken = time.time()-t_start
    logFile.add(f"All gene paths calculated between {len(genes)} genes ({int(np.power(len(genes), 2))} combinations); took {timeTaken/60:.2f} minutes, i.e. {timeTaken/3600:.2f} hours")

    return

def pathCheckWorker(threadSimple: int, toCheck: list, genes: list, saveDir: str, graphSTRING: ig.Graph, logFile: Logger):
    logFile.add(f"Thread {threadSimple} initialised")
    t_parallelStart = time.time()
    for geneBase in toCheck:
        paths = {}
        for geneTarget in genes:
            # Check if this path has already been calculated; if so use it to fill in this inverse path
            if(os.path.exists(os.path.join(saveDir, f"{geneTarget}_Paths.json"))):
                with open(os.path.join(saveDir, f"{geneTarget}_Paths.json"), "r") as f:
                    targetPaths = json.load(f)
                # Get relevant path
                relPath = targetPaths[geneBase]
                # Set up dictionary
                # Path nodes are added in this way so the output JSON will have the nodes in the correct order
                pathnodes: dict = {"path length": relPath["path length"]}
                for i in range(relPath["path length"]+1):
                    pathnodes[f"Path node {i}"] = ""
                # Go through other nodes and add them to this path inverted (so we're calculating the path B -> A from path A -> B)
                for key in relPath:
                    # Get what node number this is in the original path
                    if("node" in key):
                        nodeNo = int(key.split(" ")[-1])
                        # Otherwise, invert the node number (i.e. path length minus this node's number)
                        newNum = int(relPath["path length"]) - nodeNo
                        pathnodes[f"Path node {newNum}"] = {"name": relPath[key]["name"]}
            else:
                # If this path has not already been calculated, do so
                shortpath = graphSTRING.get_shortest_paths(geneBase, to=geneTarget, weights=graphSTRING.es["combined_score"], output="vpath")[0]
                # Convert these into a list of nodes - by default it has start, [intermediates], stop as the results
                # So the first and last entry COULD BE removed for space efficiency as the starting and stopping points are known
                pathnodes_raw = [graphSTRING.vs[v] for v in shortpath]
                # Format this path into a more readable/usable form
                pathnodes: dict = {"path length": len(pathnodes_raw) - 1}
                for i in range(len(pathnodes_raw)):
                    pathnodes[f"Path node {i}"] = {"name": pathnodes_raw[i].attributes()["name"]}
            paths[geneTarget] = deepcopy(pathnodes)
        with open(os.path.join(saveDir, f"{geneBase}_Paths.json"), "w") as f:
            json.dump(paths, f, indent = 4)
        logFile.add(f"Thread {threadSimple} finished gene {geneBase}")
    timeTaken = time.time()-t_parallelStart
    logFile.add(f"Thread {threadSimple} finished; took {timeTaken/60:.2f} minutes, i.e. {timeTaken/3600:.2f} hours")
    return


### Legacy code
"""
def calculate_target_paths(g: ig.Graph, target: str, startPoints: list) -> dict:
    paths = {}
    for sp in startPoints:
        shortpath = g.get_shortest_paths(sp, to=target, weights=g.es["combined_score"], output="vpath")[0]
        # Convert these into a list of nodes - by default it has start, [intermediates], stop, 
        # So the last entry is removed as the stopping point (the target) is known and the same for all
        pathnodes = [g.vs[v] for v in shortpath][:-1]
        # Format this path into a more readable/usable form
        paths[sp] = deepcopy({f"Path node {i}": {"name": pathnodes[i].attributes()["name"],
                                                 "survivability": pathnodes[i].attributes()["survivability"]}} 
                                for i in range(len(pathnodes)))
        # Record path length for easy access to distance between predicted target and actual target
        paths[sp]["path length"] = len(pathnodes)
    return paths

def calculate_shortest_path(g: ig.Graph, target: str, startPoints: list) -> dict:
    shortest, shortestpath = np.inf, []
    for sp in startPoints:
        shortpath = g.get_shortest_paths(sp, to=target, weights=g.es["combined_score"], output="vpath")[0]
        if(shortest>len(shortpath)):
            shortestpath = deepcopy(shortpath)
            shortest = len(shortpath)
    # Get the nodes in this shortest path
    pathnodes = [g.vs[v] for v in shortestpath]
    return {"path length": len(pathnodes)-1,
            "path": \
                {f"Path node {i}": {"name": pathnodes[i].attributes()["name"],
                        "survivability": pathnodes[i].attributes()["survivability"]} \
                                for i in range(len(pathnodes))}}

def calculate_drug_paths(drug: str, g_base: ig.Graph, tdpfp: str, drugGeneSurv: dict, allbyallcol: pd.Series, drugTargetsRefined: pd.DataFrame,
                         coresPerProcess: Optional[int]):
    # If there are no targets or this drug's paths have already been calculated, skip
    targets = drugTargetsRefined["Target"].values
    if(len(targets)<1):
        return "No Targets"
    if(os.path.exists(os.path.join(tdpfp, f"{drug}.json"))):
        print(f"Found drug path data already calculated for {drug}; skipping...")
        return "Complete"
    #print(f"Calculating {drug}...")
    # Make sure there's at least 1 core available if it isn't a Nonetype
    if(coresPerProcess is not None):
        if(type(coresPerProcess)==int):
            coresPerProcess = max(coresPerProcess, 1)
        else:
            coresPerProcess = None
    # Set up results
    drugResults = {}
    # Get the appropriate threshold for 'starting point' genes
    survivability_cutoff = get_survivability_threshold(drug, drugGeneSurv, np.array(allbyallcol.values, dtype = float))
    # Get drug targets and save results output
    g = deepcopy(g_base)
    g.vs["survivability"] = [allbyallcol.loc[gn] if gn in allbyallcol.index else float("NaN") for gn in g.vs["name"]]
    startPoints = [n for s,n in zip(g.vs["survivability"], g.vs["name"]) if s >= survivability_cutoff]
    # Get the shortest path for each target using parallel processing or non-parallel processing if specified
    if(coresPerProcess is None):
        results = []
        for target in targets:
            results.append(calculate_target_paths(deepcopy(g), target, startPoints))
    else:
        with mp.Pool(coresPerProcess) as p:
            results = p.starmap(calculate_target_paths, [(deepcopy(g), target, startPoints) for target in targets], chunksize = int(len(targets)/coresPerProcess))
    for target, result in zip(targets, results):
        drugResults[target] = result
    with open(os.path.join(tdpfp, f"{drug}.json"), "w") as f:
        json.dump(drugResults, f)
    #print(f"Saved graph path data for {drug}")
    return
"""

if(__name__=="__main__"):
    main()