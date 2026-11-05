#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created June 2026

@author: jds40
"""

import traceback
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

from drug_search import update_hgnc, get_targets_all
from logger import Logger
from shared_functions import split_list

DEFAULT_HUGO_FILE: str = os.path.join("Data", "Laurence-Data", "hgnc_complete_set.tsv")
DEFAULT_STRING_INFO_FILE: str = os.path.join("Data", "Laurence-Data", "9606.protein.info.v12.0.txt")
DEFAULT_STRING_LINK_FILE: str = os.path.join("Data", "Laurence-Data", "9606.protein.links.full.v12.5.txt")
DEFAULT_ALLBYALL: str = os.path.join("Data", "Laurence-Data", "AllDrugsByAllGenes.tsv")

# Set whether we're in debug mode
DEBUG_MODE: bool = False
if(os.path.exists(os.path.join("Local", "localVars.json"))):
    with open(os.path.join("Local", "localVars.json")) as f:
        DEBUG_MODE: bool = json.load(f)["DEBUG_MODE"]


def main():
    translator = import_stringTranslator(stinfoLoc=DEFAULT_STRING_INFO_FILE)
    links = import_stringConnections(stlinkLoc=os.path.join("Data", "Raw Data", "STRING", "calculationSTRING-9606-v12.5-experiments_databases.tsv"),
                                     combined_min=600)
    output = analysisSTRING(translator, links, coreCount=40)
    #df = pd.read_csv(os.path.join("Data", "Raw Data", "STRING", "calculationSTRING_12.0_experiments_databases.tsv"), sep = "\t")
    #print(df["protein1"].str.lower().str.contains("ensp").unique())
    #stinfo, stlink = update_stringNames(stlinkLoc=os.path.join("Data", "Raw Data", "STRING", "calculationSTRING_12.0_experiments_databases.tsv"))
    #string_analysis()

def import_stringTranslator(stinfoLoc: str = DEFAULT_STRING_INFO_FILE,
                       hugoLoc: str = DEFAULT_HUGO_FILE, gdscLoc: str = DEFAULT_ALLBYALL) -> pd.DataFrame:
    # Load HGNC data
    hgnc = pd.read_table(hugoLoc, low_memory=False).fillna('')
    hgnc = hgnc[['symbol', 'ensembl_gene_id',
                'prev_symbol', 'location', 'location_sortable']]
    hgnc.set_index('symbol', inplace=True)

    # Fetch list of genes from GDSC to check
    dxg = pd.read_csv(gdscLoc, sep = "\t")
    genesOrig = dxg["symbol"].values
    dxg = update_hgnc(dxg, hgnc)
    genesHugo = dxg["symbol"].values
    dxg.set_index("symbol", inplace = True)

    # Get stinfo file
    stinfo = pd.read_csv(stinfoLoc, sep = "\t")
    # Refine to entries with a valid 'preferred name' and those found in humans
    sthuman = stinfo.loc[~stinfo["preferred_name"].isna()]
    sthuman = sthuman.loc[sthuman["#string_protein_id"].str.contains("9606")]

    # Get STRING ID for each GDSC gene
    toFrame = []
    for geneO, geneH in zip(genesOrig, genesHugo):
        if(geneH in sthuman["preferred_name"].values):
            toFrame.append(deepcopy((geneO, geneH, sthuman.loc[sthuman["preferred_name"]==geneH]["#string_protein_id"].values[0])))
        elif(geneO in sthuman["preferred_name"].values):
            toFrame.append(deepcopy((geneO, geneH, sthuman.loc[sthuman["preferred_name"]==geneO]["#string_protein_id"].values[0])))
        else:
            toFrame.append(deepcopy((geneO, geneH, "")))
    translator = pd.DataFrame(data = toFrame, columns = ["Original Name", "Hugo Name", "STRING ID"])

    return translator

def import_stringConnections(stlinkLoc: str = DEFAULT_STRING_LINK_FILE, hugoLoc: str = DEFAULT_HUGO_FILE,
                             combined_min: int = 600) -> pd.DataFrame:
    # Load HGNC data
    hgnc = pd.read_table(hugoLoc, low_memory=False).fillna('')
    hgnc = hgnc[['symbol', 'ensembl_gene_id',
                'prev_symbol', 'location', 'location_sortable']]
    hgnc.set_index('symbol', inplace=True)

    ## STRING network
    # Import network
    linksep = {"ssv": " ", "csv": ",", "tsv": "\t"}[stlinkLoc.split(".")[-1].lower()]
    stlink = pd.read_csv(stlinkLoc,sep=linksep)
    stlink.combined_score = stlink.combined_score.astype(int)
    # Remove non-human links
    stlink = stlink.loc[(stlink["protein1"].str.contains("9606")) & (stlink["protein2"].str.contains("9606"))]
    # Refine STRING links to those with a combined score >0.6 and return the dataframe
    stlink = stlink.loc[stlink.combined_score >= combined_min]
    return stlink.rename(columns = {"combined_score": "CombinedScore"})

def analysisSTRING(stTranslator: pd.DataFrame, stlink: pd.DataFrame,
                   gdscLoc: str = DEFAULT_ALLBYALL, logFile: Optional[Logger] = None,
                   saveDir: str = os.path.join("Data", "Results", "STRING-Analysis"),
                   coreCount: int = max(mp.cpu_count()-2, 1)):
    # Set up logging file
    rLog: bool = False
    if(logFile is not None):
        rLog = True

    # Convert the STRING connectivity information to a graph with igraph
    if(rLog):
        logFile.add('Building human link graph ...')

    g_global = ig.Graph.TupleList(stlink.itertuples(index=False), directed=False, weights=False, edge_attrs="CombinedScore")
    
    if(rLog):
        logFile.add('Done !!')
    # Make sure the graph is sufficiently connected for our purposes
    if(not g_global.is_connected()):
        if(rLog):
            logFile.add("STRING network graph is not fully connected; try lowering score requirement or removing unneeded cliques")
        print("STRING network graph is not fully connected; try lowering score requirement or removing unneeded cliques")
    
    # Make database for outputs
    if(not os.path.exists(saveDir)):
        os.mkdir(saveDir)
    if(not os.path.exists(os.path.join(saveDir, "Gene-Paths"))):
        os.mkdir(os.path.join(saveDir, "Gene-Paths"))
    if(rLog):
        logFile.add(f"Files saving to {saveDir}")

    # Load GDSC SC data
    sep = {"ssv": " ", "csv": ",", "tsv": "\t"}[gdscLoc.split(".")[-1].lower()]
    dxg = pd.read_csv(gdscLoc, sep = sep)
    dxg.set_index("symbol", inplace = True)

    # Get a list of genes which are known putative targets - these will be our start points
    targets = get_target_list()
    startPoints, badDrugs, badDrugGenes = {}, [], []
    for drug in targets:
        # Make sure drug is in GDSC data
        if(drug not in dxg.columns):
            badDrugs.append(drug)
            continue
        startPoints[drug] = []
        for gene in targets[drug]:
            if((gene in stTranslator["Original Name"].values or gene in stTranslator["Hugo Name"].values) and gene not in startPoints):
                rel = stTranslator.loc[(stTranslator["Original Name"] == gene) | (stTranslator["Hugo Name"] == gene)]
                sID = rel["STRING ID"].values[0]
                startPoints[drug].append(sID)
            elif(gene not in stTranslator["Original Name"].values and gene not in stTranslator["Hugo Name"].values):
                badDrugGenes.append(f"{drug}-{gene}")
        if(len(startPoints[drug])<1):
            del startPoints[drug]

    # Break up list of drugs/start points for parallel proessing
    toPass = []
    for drug in startPoints:
        toPass.append([startPoints[drug], dxg[drug]])

    if(DEBUG_MODE):
        coreCount = min(2, coreCount)
        toPass = toPass[:5]
    toPass = split_list(toPass, coreCount)

    # Run parallel processing
    try:
        mp.Pool(coreCount).starmap(superSTRINGworker,
                                    [(deepcopy(g_global), toPass[i], stTranslator, saveDir) for i in range(len(toPass))])

        # Load and concatenate data
        results = []
        for file in os.listdir(os.path.join(saveDir, "tempStar")):
            results.append(pd.read_csv(os.path.join(saveDir, "tempStar", file), sep = "\t", index_col = None))
        results: pd.DataFrame = pd.concat(results, ignore_index=True).reset_index()
        del results["index"]
        # Add in bad drugs and genes to the end of the lise
        badData = [(drug, "", np.nan, "", False) for drug in badDrugs] + [(e.split("-")[0], e.split("-")[1], np.nan, "", False) for e in badDrugGenes]
        results = pd.concat([results, pd.DataFrame(data = badData, columns = results.columns)], ignore_index=True).reset_index()
        # Save new data and remove temporary data
        rDir = "Gene-Paths.tsv"
        if(DEBUG_MODE):
            rDir = "DEBUG-" + rDir
        results.to_csv(os.path.join(saveDir, rDir), sep = "\t", lineterminator="\n", index = False)
        for file in os.listdir(os.path.join(saveDir, "tempStar")):
            os.remove(os.path.join(saveDir, "tempStar", file))
        os.rmdir(os.path.join(saveDir, "tempStar"))
        return results
    except Exception as e:
        print(f"Failed to run STRING analysis: {e}.\n")
        print(traceback.format_exc())

def superSTRINGworker(network: ig.Graph, batchPart: list, translator: pd.DataFrame, saveDir: str):
    # Passes one form of data to STRING workers; needs to be formatted like this for better optimisation (in terms of memory requirements; slightly worse for time)
    for drug in batchPart:
        STRINGworker(deepcopy(network), drug[0], drug[1], translator, saveDir)
    return

def STRINGworker(network: ig.Graph, startPoints: list, survCorrs: pd.Series, translator: pd.DataFrame, saveDir: str):

    survCorrs = survCorrs[~np.isnan(survCorrs)]
    # First, apply survivability Correlations to network points
    for vs in network.vs:
        rel = translator.loc[translator["STRING ID"]==vs["name"]]
        if(len(rel)<1):
            continue
        orig, hugo = rel["Original Name"].values[0], rel["Hugo Name"].values[0]
        vs["Pref Name"] = orig
        vs["Hugo Name"] = hugo
        if(hugo in survCorrs.index):
            vs["SC Zscore"] = float(np.divide(survCorrs[hugo] - np.mean(survCorrs.values), np.std(survCorrs.values)))
        elif(orig in survCorrs.index):
            vs["SC Zscore"] = float(np.divide(survCorrs[orig] - np.mean(survCorrs.values), np.std(survCorrs.values)))
        else:
            vs["SC Zscore"] = np.nan

    # Set up function output
    output = []
    # Go through the different start points (which are STRING ID's) and determine distance from Z>=1.645
    for sp in startPoints:
        # Get Starting vertex
        try:
            vertex = network.vs.find(name=sp)
        except:
            output.append((survCorrs.name, name, np.nan, "NOT IN GRAPH", False))
            continue
        # Begin bredth-first search to find relevant Z-Score
        explored: dict = {}
        current_layer: list = [vertex["name"]]
        next_layer: list = []
        layerCount: int = 0
        zFound: bool = False
        while(len(current_layer) > 0):
            # Get current vertex
            cVertex = network.vs.find(current_layer[0])
            # Check if the Survivability Correlation Z-Score is sufficient
            if(cVertex["SC Zscore"] is not None):
                if(cVertex["SC Zscore"] > 1.645):
                    zFound = True
                    pathIndex = network.get_shortest_paths(sp, to = cVertex["name"], output = "vpath")[0]
                    path = [network.vs[i]["Hugo Name"] for i in pathIndex][::-1]
                    break
            # Mark that vertex as explored
            explored[cVertex["name"]] = True
            # Get all neighbours and add their index to the next layer list
            neighbours = cVertex.neighbors()
            for v in neighbours:
                if(v["name"] not in explored):
                    next_layer.append(v["name"])
            # Remove the currently-investigated node from the current layer list
            current_layer.pop(0)
            if(len(current_layer) < 1):
                if(DEBUG_MODE):
                    print(f"Moving from layer {layerCount} to {layerCount+1}")
                current_layer = deepcopy(next_layer)
                next_layer = []
                layerCount += 1

        # Mark the results of this search

        name = translator.loc[translator["STRING ID"]==sp]["Hugo Name"].values[0]
        if(not zFound):
            output.append((survCorrs.name, name, np.nan, "", zFound))
        else:
            output.append((survCorrs.name, name, len(path) - 1, ",".join(path), zFound))

        if(DEBUG_MODE):
            output = pd.DataFrame(data = output, columns = ["Drug", "Putative Target", "Separation", "Path", "Path Found"])
            if(not os.path.exists(os.path.join(saveDir, "tempStar"))):
                os.mkdir(os.path.join(saveDir, "tempStar"))
            output.to_csv(os.path.join(saveDir, "tempStar", f"{survCorrs.name}-paths.tsv"), sep = "\t", lineterminator = "\n", index = False)
            return

    output = pd.DataFrame(data = output, columns = ["Drug", "Putative Target", "Separation", "Path", "Path Found"])
    # Save output
    if(not os.path.exists(os.path.join(saveDir, "tempStar"))):
        os.mkdir(os.path.join(saveDir, "tempStar"))
    output.to_csv(os.path.join(saveDir, "tempStar", f"{survCorrs.name}-paths.tsv"), sep = "\t", lineterminator = "\n", index = False)
    return

def update_stringNames(stinfoLoc: str = DEFAULT_STRING_INFO_FILE, stlinkLoc: str = DEFAULT_STRING_LINK_FILE, logFile: Optional[Logger] = None, dMode: bool = False):
    ## Prepare main STRING grpaph
    # Fetch HGNC/HUGO for updating gene names
    if(logFile is not None):
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
    if(dMode):
        genes = genes[:10]

    # STRING proteins
    if(logFile is not None):
        logFile.add("Importing STRING data")
    stinfo =  pd.read_table(stinfoLoc,
                            usecols=['#string_protein_id','preferred_name']).fillna('')
    stinfo.rename(columns = {'#string_protein_id':'ID','preferred_name':'symbol'},inplace=True)
    stinfo = update_hgnc(stinfo, hgnc)

    # Link STRING ID's and HUGO-corrected Gene names
    stdict = dict(zip(stinfo.ID,stinfo.symbol))

    # STRING network
    linksep = {"ssv": " ", "csv": ",", "tsv": "\t"}[stlinkLoc.split(".")[-1].lower()]
    stlink = pd.read_csv(stlinkLoc,sep=linksep)
    stlink.combined_score = stlink.combined_score.astype(int)
    # Change proteins in stlink from ID's to HGNC-checked names
    stlink.protein1 = stlink.protein1.apply(lambda x: stdict[x])
    stlink.protein2 = stlink.protein2.apply(lambda x: stdict[x])
    # Refine STRING links to those with a combined score >0.6
    stlink[stlink.combined_score.gt(600)]

    return stinfo, stlink

def string_analysis(saveDir: str = os.path.join("Data", "Results", "Target-Analysis"), logFile: Optional[Logger] = None,
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
    stinfo, stlink = update_stringNames(logFile=logFile, dMode = DEBUG_MODE)
    
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

    # Get a list of genes which are known putative targets - these will be our start points
    targets = get_target_list()
    startPoints = []
    for drug in targets:
        for gene in targets[drug]:
            if(gene in genesString and gene not in startPoints):
                startPoints.append(gene)

    # Get appproximation of number of combinations if all genes are valid
    approxcombs: int = 0
    for i in range(len(startPoints)):
        approxcombs += len(genes) - i

    logFile.add(f"Building list of combinations to find pathways between (approximately {approxcombs} combinations to calculate)")
    t_base = time.time()
    # Construct a list of combinations of genes to check
    combs, done = [], {}
    badGenes: dict = {}
    for startPoint in startPoints:
        # Ensure gene is valid
        if(startPoint not in genesString):
            badGenes[startPoint] = True
            continue
        for geneEnd in genes:
            # If this combination has already been calculated, skip it
            if(geneEnd in done):
                continue
            # Ensure gene is valid
            if(geneEnd not in genesString):
                badGenes[geneEnd] = True
                continue
            combs.append(deepcopy((startPoint, geneEnd)))
        # Add this gene to list of finished genes to remove duplicate work
        done[startPoint] = True
    
    logFile.add(f"{len(badGenes)} genes were found in GDSC which are not within the STRING database. Saving to {os.path.join(saveDir, 'bad_genes.txt')}")
    with open(os.path.join(saveDir, "bad genes.txt"), "w") as f:
        f.write("\n".join(list(badGenes.keys())))
    logFile.add(f"Calculating shortest paths for {len(combs)} combinations of valid genes")
    
    # Split up list of combinations into (roughly) equally-sized sub-lists which will all be passed to parallel workers
    batchList: list = split_list(combs, coreCount)

    # Pass these combinations of genes to parallel workers to get (collectively) all pathways
    mp.Pool(coreCount).starmap_async(pathCheckWorker,
                [(i, batchList[i], g_global, os.path.join(saveDir, "Gene-Paths"), 3600, logFile)
                for i in range(coreCount)]).get()
    #pathCheckWorker(0, batchList[i], g_global, os.path.join(saveDir, "Gene-Paths", 3600, logFile))

    t_taken = time.time() - t_base
    logFile.add(f"Saved {len(combs)} calculated pathways between {len(genes)-len(badGenes)} valid genes; took {t_taken/60:.2f} minutes, i.e. {t_taken/3600:.1f} hours")
    
    # Coallate these calculations so each gene has a single json file with all of its pathways
    logFile.add(f"Coallating results")
    t_base = time.time()

    goodGenes = [gene for gene in genes if gene not in badGenes]
    batchList = split_list(startPoints, coreCount)
    mp.Pool(coreCount).starmap_async(pathCoallateWorker,
                [(i, batchList[i], goodGenes, os.path.join(saveDir, "Gene-Paths"))
                for i in range(coreCount)]).get()
    #pathCoallateWorker(1, [x for xs in batchList for x in xs], goodGenes, os.path.join(saveDir, "Gene-Paths"))
    
    t_taken = time.time()-t_base
    logFile.add(f"Results coallated ({t_taken/60:.2f} minutes). Deleting raw files.")

    for geneBase in startPoints:
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

def pathCheckWorker(threadSimple: int, combinations: list[tuple], graphSTRING: ig.Graph, saveDir: str,
                    checkTimeSeconds: int = 3600, logFile: Logger = Logger("pathCheckWorkerLog.txt")) -> None:
    t_base, t_prev, fin = time.time(), time.time(), len(combinations)
    for iComb, comb in enumerate(combinations):
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
        
        # Report progress once checkTimeSeconds has been reached
        if(threadSimple == 1):
            if(time.time()-t_prev > checkTimeSeconds or iComb==0):
                t_taken, completePercent = time.time()-t_base, (iComb/fin)*100
                logFile.add(f"Thread {threadSimple} {completePercent:.2f}% Complete ({t_taken/3600:.1f} hours taken total)")
                t_prev = time.time()

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
        if(f"{geneBase}-{geneBase}.json" in destinationFiles):
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

def get_target_list(drugTargets: Optional[pd.DataFrame] = None):
    if(drugTargets is None):
        drugTargets = get_drugTargets()
    # Dictionary output used in form {drug1 : [drug 1 target 1, drug 1 target 2, ...], drug 2: [...], ...}
    results: dict = {}
    for drug, gene in zip(drugTargets["DRUG"].values, drugTargets["TARGET"].values):
        if(drug not in results):
            results[drug] = []
        if(gene not in results[drug] and gene != ""):
            results[drug].append(gene)
    return results

def get_drugTargets() -> pd.DataFrame:
    """Get all known putative drug targets

    Returns:
        pd.DataFrame: _description_
    """
    # Go through PubChem identifiers
    pubchemchembl = pd.read_csv(os.path.join("Data", "Derived-Data", "pubchem-chembl.tsv"), sep = "\t")

    pubchemchembl.dropna(inplace = True)
    pubchemchembl.drop_duplicates("PubChem", inplace=True)

    check = get_targets_all()

    # Get unidentifiable compounds
    nonan = check.dropna(axis = "index", how = "all", inplace = False)
    help = []
    for dn in check.index:
        if(dn not in nonan.index):
            help.append(dn)

    # Save unidentifiable compounds to tsv file, or append them if one already exists
    udp = os.path.join("Data", "Derived-Data", "unknown_drugs.tsv")
    if(os.path.exists(udp)):
        df = pd.read_csv(udp, sep = "\t")
    else:
        df = pd.DataFrame(data = None, columns = ["Drug", "Gene Targets"])
    toadd = [h for h in help if h not in df["Drug"].values]
    toadd = pd.DataFrame(data = zip(toadd, [np.nan for _ in range(len(toadd))]), columns = ["Drug", "Gene Targets"])
    output = pd.concat((df, toadd))
    output.to_csv(udp, sep = "\t", lineterminator="\n", index = False)

    internals = output.loc[output["Alternate Names"]=="INTERNAL COMPOUND"]["Drug"]
    internals = {i: True for i in internals}
    check["Internal Compound"] = check.index.map(internals)

    # Refine the DataFrame so we've just got a list of drugs and targets
    drugTargets = check.loc[check["Internal Compound"]!=True]
    del drugTargets["Internal Compound"]

    # Get all relevant entries from drugTargets
    targets = {}
    for row in drugTargets.iterrows():
        drug = row[0]
        entry = row[1]
        new = []
        # drop NaN entries
        entry = entry.dropna()
        # Get all relevant values
        try:
            for i in entry.index:
                # Get just the gene name targets as lists and add them to target list
                if(i=="DrugBank"):
                    new += list(entry[i].keys())
                else:
                    if(type(entry[i]) == list):
                        tosplit = ",".join(entry[i])
                    elif(type(entry[i]) == str):
                         tosplit = entry[i]
                    else:
                        tosplit = ",".join(entry[i].tolist())
                    tosplit = tosplit.replace(" ","").replace("[","").replace("]","").replace("\"","")
                    new += tosplit.split(",")
            targets[drug] = deepcopy(new)
        except Exception as e:
            print(f"Error with entry {entry}: {e}")
            break
    
    data = []
    for drug in targets.keys():
        for i in range(len(targets[drug])):
            data.append((drug, targets[drug][i]))

    drugTargets = pd.DataFrame(data, columns = ["DRUG", "TARGET"])
    drugTargets.drop_duplicates(inplace = True)
    drugTargets["TARGET"] = drugTargets["TARGET"].str.replace("'","")
    return drugTargets

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