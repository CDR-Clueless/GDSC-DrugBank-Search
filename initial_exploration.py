#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2025

@author: jds40
"""

import os
import numpy as np
import pandas as pd

from typing import Optional

from drug_search import Searcher

CLEAN_DATA_DIR: str = os.path.join("Data", "Laurence-Data")

def main():
    #targets = get_targets("somegene")
    df = pd.read_csv(os.path.join(CLEAN_DATA_DIR, "TargetRanking.tsv"), sep = "\t")
    test = df["DRUG"].unique()[0]
    searcher = Searcher()
    for drug in df["DRUG"].unique():
        targets = searcher.get_targets(drug)
        if(len(targets)>0):
            break
    print(f"{drug} targets: {targets}")
    #print(df.loc[df["DRUG"]==test])
    #print(Searcher().get_targets(test))
    return

if __name__ == "__main__":
    main()
