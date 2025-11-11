#./.venv/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 11 Nov 2025

@author: jds40
"""

from copy import deepcopy
import os
import pandas as pd
from lxml import etree
from typing import Union

class DataHandler:
    def __init__(self, datasets: Union[tuple, list, None] = None):
        # Setup internal datasets list
        self.datasets = {}
        if(datasets is None):
            return
        # Check if the datasets provided is a matrix or an array (tuple of tuples i.e. multiple datasets or just an array i.e. 1 dataset) and load as appropriate
        if(len(datasets)==2):
            if(type(datasets[0]) not in [tuple, list]):
                self.load_data(datasets[0], datasets[1])
                return
        for pair in datasets:
            self.load_data(pair[0], pair[1])
        return
    
    def load_data(self, dataset_id: str, dataset_fileloc: str) -> None:
        # First, check that the current dataset_id is not already in use
        while(dataset_id in self.datasets):
            print(f"Data ID {dataset_id} is already in use. Renaming to {dataset_id+'-c'}")
            dataset_id += "-c"
        # Next, load dataset if possible
        if(os.path.exists(dataset_fileloc)==False):
            print(f"No file found at {dataset_fileloc}. Cannot load data.")
            return
        fileend = dataset_fileloc.split(".")[-1].lower()
        if(fileend=="csv"):
            data = pd.read_csv(dataset_fileloc)
        elif(fileend=="tsv"):
            data = pd.read_csv(dataset_fileloc, sep = "\t")
        elif(fileend in ["xls", "xlsx"]):
            data = pd.read_excel(dataset_fileloc)
        elif(fileend == ".xml"):
            with open(dataset_fileloc, "r") as f:
                data = etree.parse(f)
        else:
            print(f"File type '.{fileend}' not recognised. Please use a supported file type.")
            return
        # Now, insert into the DataHandler datasets dictionary
        self.datasets[dataset_id] = deepcopy(data)
        return
    
    def erase_data(self, dataset_id: str) -> None:
        if(dataset_id not in self.datasets):
            print(f"Data ID '{dataset_id}' not found.")
            return
        del self.datasets[dataset_id]
    
    def read_data(self, dataset_id: str):
        if(dataset_id not in self.datasets):
            print(f"Data ID '{dataset_id}' not found.")
            return
        return self.datasets[dataset_id]
    
    def insert_data(self, dataset_id: str, data: Union[pd.DataFrame, etree.ElementTree], overwrite: bool = False, copy: bool = False) -> None:
        # Check whether the dataset_id provided is already loaded into the DataHandler and whether to overwrite if so
        if(dataset_id in self.datasets and overwrite==False):
            print(f"Data ID {dataset_id} already exists. Please erase this dataset first or set overwrite to True")
            return
        elif(dataset_id in self.datasets and overwrite):
            print(f"Data ID {dataset_id} already exists. Overwriting...")
        # Insert data into DataHandler
        if(copy):
            self.datasets[dataset_id] = deepcopy(data)
        else:
            self.datasets[dataset_id] = data
        print(f"Inserted data with ID {dataset_id}")
        return