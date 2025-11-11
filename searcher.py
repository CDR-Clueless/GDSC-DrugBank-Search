#./.venv/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 11 Nov 2025

@author: jds40
"""

import os
import pandas as pd
from typing import Union, Optional

from data_handler import DataHandler

from drug_search import get_data, get_targets

class Searcher(DataHandler):
    def __init__(self):
        super().__init__()
        # Load data so it doesn't later need re-loading
        db, g1, g2 = get_data()
        self.insert_data("DrugBank", db)
        self.insert_data("GDSC1", g1)
        self.insert_data("GDSC2", g2)
        print("Searcher successfully initialised")
        return
    
    def get_targets(self, drug: str):
        # Pass DB/G1/G2 databases so they don't need reloading
        return get_targets(drug, (self.datasets["DrugBank"], self.datasets["GDSC1"], self.datasets["GDSC2"]))