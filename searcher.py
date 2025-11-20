#./.venv/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 11 Nov 2025

@author: jds40
"""

import os
import pandas as pd
from typing import Union, Optional
import json
from tqdm import tqdm
import ast

from data_handler import DataHandler

from drug_search import get_data, get_targets, update_hgnc

class Searcher(DataHandler):
    def __init__(self, include_tr: bool = False, clean_datadir: str = os.path.join("Data", "Laurence-Data")):
        super().__init__()
        # Load data so it doesn't later need re-loading
        self.insert_data("HUGO", pd.read_table(os.path.join(clean_datadir, "hgnc_complete_set.tsv"), low_memory=False).fillna(''))
        db, g1, g2 = get_data()
        # Update GDSC1/2 data using HUGO database
        g1, g2 = update_hgnc(g1, self.datasets["HUGO"], column = "DRUG_NAME"), update_hgnc(g2, self.datasets["HUGO"], column = "DRUG_NAME")
        self.insert_data("DrugBank", db)
        self.insert_data("GDSC1", g1)
        self.insert_data("GDSC2", g2)
        # Add drug ranking data from LP code if appropriate
        if(include_tr):
            self.__insert_TR(clean_datadir)
        print("Searcher successfully initialised")
        return
    
    def get_targets(self, drug: str) -> Optional[dict]:
        # Pass DB/G1/G2 databases so they don't need reloading
        return get_targets(drug, (self.datasets["DrugBank"], self.datasets["GDSC1"], self.datasets["GDSC2"]), self.datasets["HUGO"])
    
    def __insert_TR(self, clean_datadir: str = os.path.join("Data", "Laurence-Data")) -> None:
        self.load_data("TargetRanking", os.path.join(clean_datadir, "TargetRanking.tsv"))
        self.load_data("GenesByDrugs", os.path.join(clean_datadir, "AllGenesByAllDrugs.tsv"))
        self.datasets["GenesByDrugs"] = self.datasets["GenesByDrugs"].rename(columns = {"Unnamed: 0": "drug"})
        return
    
    def gen_rankings(self, return_variable: bool = True, save_json: bool = True,
                     clean_datadir: str = os.path.join("Data", "Laurence-Data")) -> Optional[dict]:
        """
        
        Generate json output of drug-gene correlation scores from GDSC and DrugBank targets using the TargetRanking code

        Args:
            return_variable (bool, optional): _description_. Defaults to True.
            save_json (bool, optional): _description_. Defaults to True.

        Returns:
            Optional[dict]: _description_
        """
        # Make sure the relevant data is loaded
        if("TargetRanking" not in self.datasets or "GenesByDrugs" not in self.datasets):
            self.__insert_TR(clean_datadir)
        # Set up output dictionary and fetch dataframe of target rankings for each gene response to each drug/target gene, and correlation scores per drug
        results: dict = {}
        df: pd.DataFrame = self.datasets["TargetRanking"]
        trScores: pd.DataFrame = self.datasets["GenesByDrugs"]
        # Go through each drug in the target rankings dataframe
        for drug in tqdm(df["DRUG"].unique(), desc = "Generating target rankings for known GDSC drugs"):
            # Get scores for each target gene of this drug
            scores_local = trScores.loc[trScores["drug"]==drug]
            targets_lp_local = list(df.loc[df["DRUG"]==drug]["TARGET"].values)
            results[drug] = {"TargetRanking Targets": {gene: {"locus": "Unknown", "score": scores_local[gene].values[0]} for gene in targets_lp_local}}
            # Get scores for the top 10 genes which responded to this drug
            targets_lbtheory = df.loc[df["DRUG"]==drug]["TOP10"].values[0]
            targets_lbtheory = ast.literal_eval(targets_lbtheory)
            results[drug]["TargetRanking Top10"] = {gene: {"locus": "Unknown", "score": scores_local[gene].values[0]} for gene in targets_lbtheory}
            # Get scores for all target genes according to DrugBank
            targets_db = self.get_targets(drug)
            results[drug]["DrugBank"] = {gene: {"locus": targets_db[gene], "score": (scores_local[gene].values[0] if gene in scores_local.columns else "Unknown")}\
                                        for gene in targets_db}
        # Save the results as a json if desired
        if(save_json):
            if(os.path.exists(os.path.join("Data", "Results"))==False):
                os.mkdir(os.path.join("Data", "Results"))
            with open(os.path.join("Data", "Results", "Laurnce-DrugBank comparison.json"), "w") as f:
                json.dump(results, f, indent = 4)
        # Return the results if desired
        if(return_variable):
            return results
        return