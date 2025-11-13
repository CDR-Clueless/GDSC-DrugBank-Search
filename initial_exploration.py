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
import seaborn as sns
from copy import deepcopy
import json
from tqdm import tqdm
import ast

from typing import Optional

from searcher import Searcher
from drug_gene_correlation_histograms import CorrelationPlotter

CLEAN_DATA_DIR: str = os.path.join("Data", "Laurence-Data")

def main():
    plotter = CorrelationPlotter()
    plotter.plot_all()
    return

if __name__ == "__main__":
    main()
