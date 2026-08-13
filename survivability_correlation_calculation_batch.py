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
    for method, nCom in zip(methods, components):
        gdsc(logFile = Logger(os.path.join("Data", "Results", "Survivability-Correlation", f"{method}-{nCom}-calcLog.log")),
             scMode = method, nComponents = nCom)
    return

if(__name__=="__main__"):
    main()
