#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 5 Nov 2026

@author: jds40
"""

import os

import pandas as pd

DEFAULT_HUGO_FILE: str = os.path.join("Data", "Laurence-Data", "hgnc_complete_set.tsv")
DEFAULT_ALLBYALL: str = os.path.join("Data", "Results", "Survivability-Correlations", "GDSC", "pIC50-pearson-AllDrugsByAllGenes.tsv")

from drug_search import update_hgnc

def main():
    return

def import_allbyall(loc: str = DEFAULT_ALLBYALL, hugoLoc: str = DEFAULT_HUGO_FILE) -> pd.DataFrame:
    """Import Drug by Gene data, making sure gene names are updated with HUGO gene standardisation

    Args:
        loc (str, optional): _description_. Defaults to DEFAULT_ALLBYALL.
        hugoLoc (str, optional): _description_. Defaults to DEFAULT_HUGO_FILE.

    Returns:
        pd.DataFrame: _description_
    """
    # Import hugo file
    hgnc = pd.read_table(hugoLoc, low_memory=False).fillna('')
    hgnc = hgnc[['symbol', 'ensembl_gene_id',
                'prev_symbol', 'location', 'location_sortable']]
    hgnc.set_index('symbol', inplace=True)

    # Import Drug by Gene data and update with HUGO data
    dxg = pd.read_csv(loc, sep = "\t")
    dxg = update_hgnc(dxg, hgnc)
    dxg.set_index("symbol", inplace=True)
    return dxg

if(__name__=="__main__"):
    main()
