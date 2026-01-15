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
import tarfile, gzip
import requests
from bs4 import BeautifulSoup as bsoup

from typing import Optional

from data_handler import DataHandler
from searcher import Searcher
from drug_gene_correlation_histograms import CorrelationPlotter, curve_guess
from drug_search import update_hgnc, get_data
from modality_analysis import ModalityAnalyzer, get_survivability_threshold

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

def calculate_target_path(g: ig.Graph, target: str, startPoints: list) -> dict:
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

def calculate_drug_paths(drug: str, g: ig.Graph, tdpfp: str, drugGeneSurv: dict, allbyallcol: pd.Series, drugTargetsRefined: pd.DataFrame,
                         coresPerProcess: int):
    print(f"Calculating {drug}...")
    # Make sure there's at least 1 core available
    coresPerProcess = max(coresPerProcess, 1)
    # Check if data for this drug already exists
    if(os.path.exists(os.path.join(tdpfp, f"{drug}.json"))):
        print(f"Found drug path data already calculated for {drug}; skipping...")
        return "Complete"
    drugResults = {}
    # Get the appropriate threshold for 'starting point' genes
    survivability_cutoff = get_survivability_threshold(drug, drugGeneSurv, np.array(allbyallcol.values, dtype = float))
    # Get drug targets and save results output
    targets = drugTargetsRefined["TARGET"].values
    g.vs["survivability"] = [allbyallcol.loc[gn] if gn in allbyallcol.index else float("NaN") for gn in g.vs["name"]]
    startPoints = [n for s,n in zip(g.vs["survivability"], g.vs["name"]) if s >= survivability_cutoff]
    # Get the shortest path for each target using parallel processing
    with mp.Pool(coresPerProcess) as p:
        results = p.starmap(calculate_target_path, [(deepcopy(g), target, startPoints) for target in targets], chunksize = int(len(targets)/coresPerProcess))
    for target, result in zip(targets, results):
        drugResults[target] = result
    with open(os.path.join(tdpfp, f"{drug}.json"), "w") as f:
        json.dump(drugResults, f)
    print(f"Saved graph path data for {drug}")

def get_cosmic_columns(cosdir: str = os.path.join("Data", "Raw Data", "COSMIC"),
                       outdir: str = os.path.join("Data", "Results", "COSMIC-columns.txt")):
    if(os.path.exists(outdir)==True):
        print(f"Loading columns...")
        with open(outdir, "r") as f:
            data = f.read()
        entries = data.split("\n\n")
        for i in range(len(entries))[::-1]:
            entries[i] = entries[i].split("\n")
            for j in range(len(entries[i]))[::-1]:
                if(len(entries[i][j].replace(" ","").replace("\t",""))==0):
                    entries[i].pop(j)
            if(len(entries[i])<2):
                entries.pop(i)
        return entries
    # Go through COSMIC archive
    columns = []
    for folder in tqdm(os.listdir(cosdir), desc = "Parsing COSMIC databases"):
        for filename in os.listdir(os.path.join(cosdir, folder)):
            if("readme" not in filename.lower()):
                #print(f"Reading {filename}")
                if(filename.lower()[-4:]==".tsv"):
                    for df in pd.read_csv(os.path.join(cosdir, folder, filename), sep = "\t", low_memory = False, chunksize = 10000, iterator = True):
                        cols = df.columns.values
                        break
                elif(filename.lower()[-4:]==".vcf"):
                    with open(os.path.join(cosdir, folder, filename), "r") as f:
                        for line in f.readlines():
                            if(line[0]=="#" and line[1]!="#"):
                                sub = line[1:].replace("\n","")
                                cols = sub.split("\t")
                        break
                else:
                    print(f"Unrecognised file type: {filename}")
                break
        columns.append(f"{filename}\n{cols}\n\n")
    with open(outdir, "w") as f:
        f.write("\n".join(columns))
    print("Columns generated.")
    return get_cosmic_columns(cosdir, outdir)

def get_cosmic_drugs(cosdir: str = os.path.join("Data", "Raw Data", "COSMIC")):
    cols = get_cosmic_columns(cosdir)
    refined = []
    for tup in cols:
        for col in tup:
            if("drug_name" in col.lower()):
                refined.append(deepcopy(tup))
    # NOTE: NEED TO FIGURE OUT HOW TO GET TO THE REFINED FILE RATHER THAN JUST HARD CODING THE LOCATION
    df = pd.read_csv(os.path.join(cosdir, "Cosmic_ResistanceMutations_Tsv_v102_GRCh38", "Cosmic_ResistanceMutations_v102_GRCh38.tsv"), sep = "\t")
    return df["DRUG_NAME"].unique()

## Drug name analysis code
def check_drug_names() -> None:
    # Get drug names listed in the COSMIC database
    cosds = get_cosmic_drugs()
    # Drug names and DrugBank names
    with open(os.path.join("Data", "Results", "correlationDrugs.txt"), "r") as f:
        corrDrugs = f.read().split("\t")[0].split("\n")
    with open(os.path.join("Data", "Results", "drugbankDrugs.txt"), "r") as f:
        dbDrugs = f.read().split("\t")[0].split("\n")
    
    total, totalDrugbank, totalCosmic, totalUnfound = 0, 0, 0, 0

    for cd in corrDrugs:
        total += 1
        check, equiv = cd.lower().replace(" ", ""), False

        for dbd in dbDrugs:
            dbcheck = dbd.lower().replace(" ", "")
            if(check==dbcheck):
                equiv = True
                totalDrugbank += 1
                break
        for cd in cosds:
            cdcheck = cd.lower().replace(" ","")
            if(check==cdcheck):
                equiv = True
                totalCosmic += 1
                break
        
        if(not equiv):
            totalUnfound += 1
    
    print(totalUnfound)
    print(total)
    print(totalDrugbank)
    print(totalCosmic)
    return

# Output drug targets
def fetch_drug_targets(save_output: bool = False) -> dict:
    ## Import relevant datasets and amend them
    # HGNC
    hgnc = pd.read_table(DEFAULT_HUGO_FILE, low_memory=False).fillna('')
    hgnc = hgnc[['symbol', 'ensembl_gene_id',
                'prev_symbol', 'location', 'location_sortable']]
    hgnc.set_index('symbol', inplace=True)

    # DrugxGene survivability scores
    allbyall = pd.read_csv(DEFAULT_ALL_BY_ALL_FILE, sep = "\t")
    allbyall = update_hgnc(allbyall, hgnc)
    allbyall = allbyall.set_index("symbol")

    # Dictionary of results for drug-gene survivability distributions
    with open(os.path.join("Data", "Results", "Drug-gene correlation frequency histograms", "stats.json"), "r") as f:
        drugGeneSurv: dict = json.load(f)

    # Fetch targets for each drug
    if(os.path.exists(os.path.join("Data", "Results", "Drug Targets"))==False):
        os.mkdir(os.path.join("Data", "Results", "Drug Targets"))
    results: dict = {}
    for drug in allbyall.columns:
        rel = allbyall[drug]
        thresh = get_survivability_threshold(drug, drugGeneSurv, survivability_array=np.array(rel.values))
        genes = np.array(rel.index)[np.array(rel.values) >= thresh]
        results[drug] = list(genes)
        if(save_output):
            with open(os.path.join("Data", "Results", "Drug Targets", f"{drug}.txt"), "w") as f:
                f.write("\n".join(genes))

    if(save_output):
        with open(os.path.join("Data", "Results", "DrugTargets.json"), "w") as f:
            json.dump(results, f, indent=4)

    return results

def main():
    # Perform initial setup
    initial_setup()
    # Load credentials and get available core count
    with open(os.path.join("Local", "passwords.json"), "r") as f:
        creds = json.load(f)
    coreCount = creds["core-count"]
    if(str(coreCount).strip().lower()=="auto"):
        coreCount = max(mp.cpu_count()-2, 1)
    else:
        coreCount = max(int(coreCount), 1)
    print(f"Using {coreCount} cores")
    
    ## Import relevant datasets and amend them
    # HGNC
    hgnc = pd.read_table(DEFAULT_HUGO_FILE, low_memory=False).fillna('')
    hgnc = hgnc[['symbol', 'ensembl_gene_id',
                'prev_symbol', 'location', 'location_sortable']]
    hgnc.set_index('symbol', inplace=True)

    # DrugxGene survivability scores
    allbyall = pd.read_csv(DEFAULT_ALL_BY_ALL_FILE, sep = "\t")
    allbyall = update_hgnc(allbyall, hgnc)
    allbyall = allbyall.set_index("symbol")

    # Dictionary of results for drug-gene survivability distributions
    with open(os.path.join("Data", "Results", "Drug-gene correlation frequency histograms", "stats.json"), "r") as f:
        drugGeneSurv: dict = json.load(f)

    # Fetch targets for each drug
    if(os.path.exists(os.path.join("Data", "Results", "Drug Targets"))==False):
        os.mkdir(os.path.join("Data", "Results", "Drug Targets"))
    results: dict = {}
    for drug in allbyall.columns:
        rel = allbyall[drug]
        thresh = get_survivability_threshold(drug, drugGeneSurv, survivability_array=np.array(rel.values))
        genes = np.array(rel.index)[np.array(rel.values) >= thresh]
        results[drug] = list(genes)

    linkBase: str = "https://davidbioinformatics.nih.gov/api.jsp?type=GENETYPE&ids=GENEIDS&tool=DAVID_TOOL&annot=ANNOTATIONCATEGORIES"

    geneType: str = "OFFICIAL_GENE_SYMBOL"
    tool: str = "summary"
    annotationTypes: list = ["all"]

    for drug in results:
        
        if(annotationTypes[0]=="all"):
            link: str = linkBase[:linkBase.index("&annot")]
        else:
            link: str = linkBase.replace("ANNOTATIONCATEGORIES", ",".join(annotationTypes))
        
        link = link.replace("GENETYPE", geneType)
        link = link.replace("DAVID_TOOL", tool)
        link = link.replace("GENEIDS", ",".join(results[drug]))

        response = requests.get(link)
        soup = bsoup(response.text, "html.parser")
        print(soup.prettify())
        break
    return


    """
    ##Modality analysis plotting code
    az = ModalityAnalyzer()
    #az.plot_cf()
    #az.plot_high_survivors()
    #az.plot_waterfall()
    az.plot_compare_targets()
    #"""

    """
    ## Target pathfinding code
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
        drugGeneSurv: dict = json.load(f)

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
    with mp.Pool(max(int(coreCount/2), 1)) as p:
        p.starmap(calculate_drug_paths, [(drug, deepcopy(g_global), tdpfp, drugGeneSurv, allbyall[drug], \
                                         drugTargets.loc[drugTargets["DRUG"]==drug], int(coreCount/2)) \
                                            for drug in allbyall.columns],
                                         chunksize=int(len(allbyall.columns)/int(coreCount/2)))
    
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
    """
    return

if __name__ == "__main__":
    main()
