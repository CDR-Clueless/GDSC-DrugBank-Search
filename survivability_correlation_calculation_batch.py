#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 11 Aug 2026

@author: jds40
"""

import os

from survivability_correlation_calculation import gdsc
from logger import Logger

def main():
    methods = ["pearson"] + (["gls"]*5) + (["wlsp"]*5) + (["wlsd"]*5)
    components = [1] + (list(range(1, 6))*3)
    outDir = os.path.join("Data", "Results", "Survivability-Correlations", "GDSC")
    if(not os.path.exists(outDir)):
        os.mkdir(outDir)
    for method, nCom in zip(methods, components):
        # Check if file already exists; skip if so
        if(os.path.exists(os.path.join(outDir, f"pIC50-{method}-AllDrugsByAllGenes.tsv"))):
            continue
        print(f"Calculating SC scores using method {method} with {nCom} components")
        gdsc(logFile = Logger(os.path.join("Data", "Results", "Survivability-Correlations", f"{method}-{nCom}-calcLog.log")),
             scMode = method, nComponents = nCom)
        print(f"Finished calculation for {method} with {nCom} components")
    return

if(__name__=="__main__"):
    main()
