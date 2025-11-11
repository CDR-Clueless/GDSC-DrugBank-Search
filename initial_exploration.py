#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2025

@author: jds40
"""

import os
import numpy as np
import pandas as pd
from copy import deepcopy
import json
from tqdm import tqdm
import ast

from typing import Optional

from searcher import Searcher

CLEAN_DATA_DIR: str = os.path.join("Data", "Laurence-Data")

def main():
    #targets = get_targets("somegene")
    df = pd.read_csv(os.path.join(CLEAN_DATA_DIR, "TargetRanking.tsv"), sep = "\t")
    scores_global = pd.read_csv(os.path.join(CLEAN_DATA_DIR, "AllGenesByAllDrugs.tsv"), sep = "\t")
    scores_global = scores_global.rename(columns = {"Unnamed: 0": "drug"})
    print(df.columns)
    searcher = Searcher()
    results = {}
    for drug in tqdm(df["DRUG"].unique()):
        scores_local = scores_global.loc[scores_global["drug"]==drug]
        targets_lp_local = list(df.loc[df["DRUG"]==drug]["TARGET"].values)
        results[drug] = {"TargetRanking Targets": {gene: {"locus": "Unknown", "score": scores_local[gene].values[0]} for gene in targets_lp_local}}
        targets_lbtheory = df.loc[df["DRUG"]==drug]["TOP10"].values[0]
        targets_lbtheory = ast.literal_eval(targets_lbtheory)
        results[drug]["TargetRanking Top10"] = {gene: {"locus": "Unknown", "score": scores_local[gene].values[0]} for gene in targets_lbtheory}
        targets_db = searcher.get_targets(drug)
        results[drug]["DrugBank"] = {gene: {"locus": targets_db[gene], "score": (scores_local[gene].values[0] if gene in scores_local.columns else "Unknown")}\
                                      for gene in targets_db}
        if(len(targets_db)>0):
            pass
            #break
    if(os.path.exists(os.path.join("Data", "Results"))==False):
        os.mkdir(os.path.join("Data", "Results"))
    with open(os.path.join("Data", "Results", "Laurnce-DrugBank comparison.json"), "w") as f:
        json.dump(results, f, indent = 4)
    return

if __name__ == "__main__":
    main()
