#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2 Jul 2026

@author: jds40
"""

import os
from copy import deepcopy

import numpy as np
import pandas as pd

from drug_search import get_targets_all

MANUAL_TARGETS: str = os.path.join("Data", "Derived-Data", "manual_targets.tsv")
CLEANED_DIR: str = os.path.join("Data", "Laurence-Data")

def main():
    get_drugTargets()

def get_drugTargets(include_manual: bool = True, include_laurence: bool = True) -> pd.DataFrame:
    """Get all known putative drug targets

    Returns:
        pd.DataFrame: DataFrame of dru targets
    """
    # Go through PubChem identifiers
    pubchemchembl = pd.read_csv(os.path.join("Data", "Derived-Data", "pubchem-chembl.tsv"), sep = "\t")

    pubchemchembl.dropna(inplace = True)
    pubchemchembl.drop_duplicates("PubChem", inplace=True)

    # Get known targets from GDSC, DrugBank and PubChem-ChEMBL data
    check = get_targets_all()

    ## Get unidentifiable compounds
    nonan = check.dropna(axis = "index", how = "all", inplace = False)
    help = []
    for dn in check.index:
        if(dn not in nonan.index):
            help.append(dn)

    ## Save unidentifiable compounds to tsv file, or append them if one already exists
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

    # Add Manually identified targets
    if(os.path.exists(MANUAL_TARGETS) and include_manual):
        extra_targets = pd.read_csv(MANUAL_TARGETS, sep = "\t")
        extra_targets.dropna(axis = "index", how = "any")
        drugTargets = pd.concat([drugTargets, extra_targets], ignore_index=True)

    # Add Targets identified by L. Pearl
    if(os.path.exists(CLEANED_DIR) and include_laurence):
        g1, g2 = os.path.join(CLEANED_DIR, "GDSC1_drug_results_target_cleaned7.tsv"), os.path.join(CLEANED_DIR, "GDSC2_drug_results_target_cleaned7.tsv")
        if(os.path.exists(g1)):
            df = pd.read_csv(g1, sep = "\t")
        if(os.path.exists(g2)):
            if(os.path.exists(g1)):
                df = pd.concat([df, pd.read_csv(g2, sep = "\t")], ignore_index=True)
            else:
                df = pd.read_csv(g2, sep = "\t")
        # Refine to relevant columns of drug name and target(s)
        df = df[["DRUG_NAME", "PUTATIVE_TARGET"]]
        # Split putative targets into individual targets and attach drug names
        out = []
        for d, pts in zip(df["DRUG_NAME"].values, df["PUTATIVE_TARGET"].values):
            pt = pts.split(",")
            out += deepcopy([[d, p] for p in pt])
        extra_targets = pd.DataFrame(data = out, columns = ["DRUG", "TARGET"]).dropna(axis = "index", how = "any")
        drugTargets = pd.concat([drugTargets, extra_targets], ignore_index=True)


    drugTargets["TARGET"] = drugTargets["TARGET"].str.replace("'","")
    drugTargets.drop_duplicates(inplace = True)
    drugTargets.reset_index(inplace = True)
    print(f"{len(drugTargets)} gene targets fetched!")
    return drugTargets

if(__name__=="__main__"):
    main()