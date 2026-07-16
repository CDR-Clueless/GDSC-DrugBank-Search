#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 16 Jul 2026

@author: jds40
"""

from GLMPrediction import import_essentiality, import_gdsc, import_linker

def main():
    ess, gdsc, linker = import_essentiality(), import_gdsc(), import_linker()
    for drug in gdsc.index.unique():
        for gene in ess.columns:
            grel = gdsc.loc[gdsc.index==drug]
            erel = ess[gene]
            cLines = set(grel["ModelID"]) & set(erel.index)
            erel = erel.loc[erel.index.isin(cLines)]
            grel = grel.loc[grel["ModelID"].isin(cLines)]
            print(grel)
            return
    return

if(__name__=="__main__"):
    main()
