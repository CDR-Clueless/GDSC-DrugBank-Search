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
    # Get list of gene names in string - dictionary is used rather than list for O(1) retrieval times
    genesString: dict = {x["name"]: True for x in g_global.vs}
    # Make database for outputs
    if(not os.path.exists(os.path.join(saveDir, "Gene-Paths"))):
        os.mkdir(os.path.join(saveDir, "Gene-Paths"))

    # Get appproximation of number of combinations if all genes are valid
    approxcombs: int = 0
    for count in range(len(genes)+1):
        approxcombs += count

    logFile.add(f"Building list of combinations to find pathways between (approximately {approxcombs} combinations to calculate)")
    t_base = time.time()
    # Construct a list of combinations of genes to check
    combs = []
    badGenes: dict = {}
    for i, geneBase in enumerate(genes):
        # Ensure gene is valid
        if(geneBase not in genesString):
            badGenes[geneBase] = True
            continue
        for geneEnd in genes[i:]:
            # Ensure gene is valid
            if(geneEnd not in genesString):
                badGenes[geneEnd] = True
                continue
            combs.append(deepcopy((geneBase, geneEnd)))
    
    logFile.add(f"{len(badGenes)} genes were found in GDSC which are not within the STRING database. Saving to {os.path.join(saveDir, 'bad_genes.txt')}")
    with open(os.path.join(saveDir, "bad genes.txt"), "w") as f:
        f.write("\n".join(list(badGenes.keys())))
    logFile.add(f"Calculating shortest paths for {len(combs)} combinations of valid genes")
    
    # Split up list of combinations into (roughly) equally-sized sub-lists which will all be passed to parallel workers
    batchList: list = split_list(combs, coreCount)

    # Pass these combinations of genes to parallel workers to get (collectively) all pathways
    mp.Pool(coreCount).starmap_async(pathCheckWorker,
                [(i, batchList[i], g_global, os.path.join(saveDir, "Gene-Paths"))
                for i in range(coreCount)]).get()
    
    t_taken = time.time() - t_base
    logFile.add(f"Saved {len(combs)} calculated pathways between {len(genes)-len(badGenes)} valid genes; took {t_taken/60:.2f} minutes, i.e. {t_taken/3600:.1f} hours")
    
    # Coallate these calculations so each gene has a single json file with all of its pathways
    logFile.add(f"Coallating results")
    t_base = time.time()

    goodGenes = [gene for gene in genes if gene not in badGenes]
    batchList = split_list(goodGenes, coreCount)
    mp.Pool(coreCount).starmap_async(pathCoallateWorker,
                [(i, batchList[i], goodGenes, os.path.join(saveDir, "Gene-Paths"))
                for i in range(coreCount)]).get()
    #pathCoallateWorker(1, goodGenes, goodGenes, os.path.join(saveDir, "Gene-Paths"))
    
    t_taken = time.time()-t_base
    logFile.add(f"Results coallated ({t_taken/60:.2f} minutes). Deleting raw files.")

    for geneBase in goodGenes:
        for geneEnd in goodGenes:
            filedir = os.path.join(saveDir, "Gene-Paths", f"{geneBase}-{geneEnd}.json")
            if(os.path.exists(filedir)):
                os.remove(filedir)
    
    t_taken = time.time() - t_start
    logFile.add(f"Raw/calculation files deleted. total time taken {t_taken/60:.2f} minutes, i.e. {t_taken/3600:.1f} hours")
    return

    ## TODO: Add code to use all pathway results to make a complete matrix
    """
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
    """
    return

def pathCheckWorker(threadSimple: int, combinations: list[tuple], graphSTRING: ig.Graph, saveDir: str) -> None:
    for comb in combinations:
        # Unpack combination to start and stop gene
        geneBase, geneTarget = comb[0], comb[1]
        # Set save directory and check if this combination's path has already been calculated
        outPath = os.path.join(saveDir, f"{geneBase}-{geneTarget}.json")
        if(os.path.exists(outPath)):
            continue
        # Get path
        shortpath = graphSTRING.get_shortest_paths(geneBase, to=geneTarget, weights=graphSTRING.es["combined_score"], output="vpath")[0]
        # Convert these into a list of nodes - by default it has start, [intermediates], stop as the results
        # So the first and last entry COULD BE removed for space efficiency as the starting and stopping points are known
        pathnodes_raw = [graphSTRING.vs[v] for v in shortpath]
        # Format this path into a more readable/usable form
        pathnodes: dict = {"path length": len(pathnodes_raw) - 1}
        for i in range(len(pathnodes_raw)):
            pathnodes[f"Path node {i}"] = {"name": pathnodes_raw[i].attributes()["name"]}

        # Save the details of this combination
        with open(outPath, "w") as f:
            json.dump(pathnodes, f)

    return

def pathCoallateWorker(threadSimple: int, genes: list[str], allGenes: list[str], saveDir: str) -> None:
    """Worker function to create formated json outputs for a list of genes

    Args:
        threadSimple (int): _description_
        genes (list[str]): _description_
        allGenes (list[str]): _description_
        saveDir (str): _description_
    """

    for geneBase in genes:
        # First, generate lists of all possible json combination files this base gene could be involved in
        originFiles, destinationFiles = [], []
        for geneNew in allGenes:
            originFiles.append(f"{geneBase}-{geneNew}.json")
            destinationFiles.append(f"{geneNew}-{geneBase}.json")
        # Remove the baseGene-baseGene duplicate
        destinationFiles.remove(f"{geneBase}-{geneBase}.json")
        # Next, refine these lists based on what files actually exist
        for i in range(len(originFiles))[::-1]:
            fileDir = os.path.join(saveDir, originFiles[i])
            if(os.path.exists(fileDir)):
                continue
            originFiles.pop(i)
        for i in range(len(destinationFiles))[::-1]:
            fileDir = os.path.join(saveDir, destinationFiles[i])
            if(os.path.exists(fileDir)):
                continue
            destinationFiles.pop(i)
        # Now, take these existing files and coallate them to make a single json detailing all pathways from this base gene
        pathways = {}
        for file in originFiles:
            with open(os.path.join(saveDir, file), "r") as f:
                pathways[file.split("-")[1].split(".")[0]] = deepcopy(json.load(f))
        for file in destinationFiles:
            with open(os.path.join(saveDir, file), "r") as f:
                reversedPath = json.load(f)
            path = {"path length": reversedPath["path length"]}
            for nodeNo in range(path["path length"]+1)[::-1]:
                newNodeNo = path["path length"] - nodeNo
                path[f"Path node {newNodeNo}"] = deepcopy(reversedPath[f"Path node {nodeNo}"])
            pathways[file.split("-")[0]] = deepcopy(path)
        
        # Export the pathways of this gene as a final file
        with open(os.path.join(saveDir, f"{geneBase}-pathways.json"), "w") as f:
            json.dump(pathways, f)
    
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