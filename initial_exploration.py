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
from urllib import request
from bs4 import BeautifulSoup as bsoup
import shutil
import time
import random

from typing import Optional, Tuple

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

def initial_setup(cosdir: str = os.path.join("Data", "Raw Data", "COSMIC")) -> None:
    """
    
    Do all necessary setup for scripts to properly run

    """
    # Ensure passwords json exists
    pdir = os.path.join("Local", "passwords.json")
    if(os.path.exists(pdir)==False):
        os.mkdir("Local")
        with open(pdir, "w") as f:
            json.dump({}, f)
    # Ensure passwords json has all necessary variables
    with open(pdir, "r") as f:
        pcont = json.load(f)
    if("core-count" not in pcont.keys()):
        pcont["core-count"] = "auto"
    with open(pdir, "w") as f:
        json.dump(pcont, f)
    # Unzip COSMIC data
    if(os.path.exists(cosdir)==False):
        print("No COSMIC directory found. Any tasks relating to COSMIC data will not function.")
    else:
        for filename in tqdm(os.listdir(cosdir), desc="Extract tar archives"):
            if(filename.lower().split(".")[-1] == "tar.gz"):
                tar = tarfile.open(os.path.join(cosdir, filename), "r:gz")
                tar.extractall(os.path.join(cosdir, filename.replace(".tar.gz","")))
                tar.close()
                os.remove(os.path.join(cosdir, filename))
        for dirname in tqdm(os.listdir(cosdir), desc="Decompressing .gz files"):
            ndir = os.path.join(cosdir, dirname)
            if(os.path.isdir(ndir)):
                for filename in os.listdir(ndir):
                    if(filename.lower().split(".")[-1]=="gz"):
                        indir = os.path.join(ndir, filename)
                        outdir = os.path.join(ndir, filename.replace(".gz",""))
                        with gzip.open(indir, "rb") as f_in:
                            with open(outdir, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        os.remove(indir)
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

def correlationDrugs_txt_check(results_dir: str = os.path.join("Data", "Results")) -> None:
    """
    Function to ensure the correlationDrugs.txt results document exists, and generate it if not
    
    :param results_dir: Directory in which to find correlationDrugs.txt
    :type results_dir: str
    """

    if(os.path.exists(os.path.join(results_dir, "correlationDrugs.txt"))==False or 
       os.path.exists(os.path.join(results_dir, "drugbankDrugs.txt"))==False):
        print("Required txt files not found; generating...")
        ModalityAnalyzer().plot_compare_targets(save_dir = os.path.join(results_dir, "modality graphs"))

## Drug name analysis code
def check_drug_names() -> Tuple[list, list, list]:
    """
    Finds which drugs listed in GDSC are in DrugBank and COSMIC
    
    :return: Drugs found in DrugBank, drugs found in COSMIC, and drugs not found in either
    :rtype: Tuple[list, list, list]
    """
    # Get drug names listed in the COSMIC database
    cosds = get_cosmic_drugs()
    # Drug names and DrugBank names
    correlationDrugs_txt_check()
    with open(os.path.join("Data", "Results", "correlationDrugs.txt"), "r") as f:
        corrDrugs = f.read().split("\t")[0].split("\n")
    with open(os.path.join("Data", "Results", "drugbankDrugs.txt"), "r") as f:
        dbDrugs = f.read().split("\t")[0].split("\n")
    
    total, totalDrugbank, totalCosmic, totalUnfound = 0, 0, 0, 0

    drugsDrugbank, drugsCosmic, drugsUnfound = [], [], []

    for cd in corrDrugs:
        total += 1
        check, equiv = cd.lower().replace(" ", ""), False

        for dbd in dbDrugs:
            dbcheck = dbd.lower().replace(" ", "")
            if(check==dbcheck):
                equiv = True
                totalDrugbank += 1
                drugsDrugbank.append(cd)
                break
        for cosd in cosds:
            cdcheck = cosd.lower().replace(" ","")
            if(check==cdcheck):
                equiv = True
                totalCosmic += 1
                drugsCosmic.append(cd)
                break
        
        if(not equiv):
            totalUnfound += 1
            drugsUnfound.append(cd)
    
    print(f"Total drugs found in DrugBank and/or COSMIC: {total}")
    print(f"Total not found in either: {totalUnfound}")
    print(f"Total drugs found in DrugBank: {totalDrugbank}")
    print(f"Total drugs found in COSMIC: {totalCosmic}")
    return (drugsDrugbank, drugsCosmic, drugsUnfound)

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

def check_drugCentral(toCheck: str) -> list | None:
    """
    Docstring for check_drugCentral
    
    :param toCheck: Drug name to check against DrugCentral records
    :type toCheck: str
    :return: list of drugs found by DrugCentral (should be in order of most likely first, least likely last)
    :rtype: list | None
    """

    # Get search result from Drug Central
    searchAddress = f"https://drugcentral.org/?q={toCheck}&approval="
    with request.urlopen(searchAddress) as response:
        html = response.read()
    # Soupify the result and find the first table (this table is what contains relevant search results)
    soup = bsoup(html, "html.parser")
    table = soup.find("table")
    # If no table was found (most likely means no results or a website error), return None
    if table is None:
        return None
    # Loop through table entries and add the names of all results (drugs) found to the list of drug names
    officialDrugs = []
    for entry in table.find_all("strong"):
        officialDrugs.append(str(entry.text).strip())
    return officialDrugs

def search_drugCentral(tofind: list, out_dir: str = os.path.join("Data", "Results", "drugCentral_dictionary.tsv")) -> None:
    # Record of new drug names found
    totalAlternatives: int = 0
    # First, find drugs already tested or make file if appropriate
    if(os.path.exists(out_dir)==False):
        with open(out_dir, "r") as f:
            f.write(f"original\talternative\n")
    preExisting: pd.DataFrame = pd.read_csv(out_dir, sep = "\t")
    toComb = [drug for drug in tofind]
    for i, drug in enumerate(preExisting["original"].values):
        if(drug in toComb):
            toComb.remove(drug)
            if(str(preExisting["alternative"].values[i]).strip().lower() not in ["none", "nan"]):
                totalAlternatives += 1
    for drug in tqdm(toComb, desc = "Searching for alternative drug names"):
        searchResults = check_drugCentral(drug)
        if(searchResults is not None):
            if(len(searchResults) > 0):
                totalAlternatives += 1
        newline = f"{drug}\t{searchResults}\n"
        with open(os.path.join("Data", "Results", "drugCentral_dictionary.tsv"), "a+") as f:
            f.write(newline)
        time.sleep(random.randint(3, 15))
    print(f"Found {totalAlternatives} new drug names out of {len(tofind)} unidentifiable GDSC drugs. Written to {out_dir}")
    return

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

    ## Use Drug Central to try and get unfound drug official names
    #toSearch = "5-AZACYTIDINE".lower()
    #print(check_drugCentral(toSearch))
    ## Get drugs not found in GDSC/COSMIC
    drugbankDrugs, cosmicDrugs, unfoundDrugs = check_drug_names()
    nondrugbank = [drug for drug in unfoundDrugs]
    for drug in cosmicDrugs:
        if(drug not in drugbankDrugs):
            nondrugbank.append(drug)
    print(f"{len(nondrugbank)} drugs in GDSC not found in DrugBank")
    # Record of new drug names found
    totalAlternatives: int = 0
    # First, find drugs already tested or make file if appropriate
    drugCentralOutput: str = os.path.join("Data", "Results", "drugCentral_dictionary.tsv")
    search_drugCentral(nondrugbank, drugCentralOutput)
    return

    ## Downloading tsv of GDSC drugs
    request.urlretrieve("https://www.cancerrxgene.org/api/compounds?list=all&sEcho=1&iColumns=7&sColumns=&iDisplayStart=0&iDisplayLength=25&mDataProp_0=0&mDataProp_1=1&mDataProp_2=2&mDataProp_3=3&mDataProp_4=4&mDataProp_5=5&mDataProp_6=6&sSearch=&bRegex=false&sSearch_0=&bRegex_0=false&bSearchable_0=true&sSearch_1=&bRegex_1=false&bSearchable_1=true&sSearch_2=&bRegex_2=false&bSearchable_2=true&sSearch_3=&bRegex_3=false&bSearchable_3=true&sSearch_4=&bRegex_4=false&bSearchable_4=true&sSearch_5=&bRegex_5=false&bSearchable_5=true&sSearch_6=&bRegex_6=false&bSearchable_6=true&iSortCol_0=0&sSortDir_0=asc&iSortingCols=1&bSortable_0=true&bSortable_1=true&bSortable_2=true&bSortable_3=true&bSortable_4=true&bSortable_5=true&bSortable_6=true&export=tsv",
                        os.path.join("Data", "Results", "GDSCdrugs.tsv"))
    """
    ### Investigate genes using online DAVID tool
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
    """

    """
    ### Modality analysis plotting code
    az = ModalityAnalyzer()
    #az.plot_cf()
    #az.plot_high_survivors()
    #az.plot_waterfall()
    az.plot_compare_targets()
    #"""

    """
    ### Target pathfinding code
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
