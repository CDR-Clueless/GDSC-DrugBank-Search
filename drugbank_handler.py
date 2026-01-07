"""
Created Dec 2025
@author: jds40
"""

import os
import pandas as pd
from lxml import etree
from drug_search import update_hgnc_single

from typing import Union

class DrugbankHandler:
    def __init__(self, dbfile: str = os.path.join("Data", "Raw Data", "full database.xml"),
                 hgncfile: str = os.path.join("Data", "Laurence-Data", "hgnc_complete_set.tsv"),
                 DrugBankPrefix: str = "{http://www.drugbank.ca}"):
        # Import DrugBank data
        with open(dbfile, "r") as f:
            db = etree.parse(f)
        self.main_tree = db
        self.dbprefix = DrugBankPrefix
        # Import HGNC data
        self.hugo = pd.read_table(hgncfile, low_memory=False).fillna('')
    
    def fetch_drug_targets(self, drug: str) -> list:
        drug = str(drug).lower().strip()
        geneTargets = {}
        # Iterate through individual drugs from drugbank
        drugBank, prefix = self.main_tree.getroot(), self.dbprefix
        for drugelement in drugBank:
            # Get the drug name and check if it is the desired drug ('drugTarget')
            namecheck = (drugelement.find(prefix+"name")).text.lower().strip()
            if(drug == namecheck or drug in namecheck):
                # Get the gene targets of this drug
                targets = drugelement.find(prefix+"targets")
                # Iterate through the gene targets, find each one's name and locus, and add this to the initial dictionary
                for target in targets.findall(prefix+"target"):
                    polypeptide = target.find(prefix+"polypeptide")
                    # If there is no polypeptide found, add this target as an exception
                    if(polypeptide is None):
                        geneTargets[f"{target} (Irregular target)"] = None
                        continue
                    genename, locus, cl = polypeptide.find(prefix+"gene-name"), polypeptide.find(prefix+"locus"), polypeptide.find(prefix+"chromosome-location")
                    # Update genename if necessary
                    genename = update_hgnc_single(genename, self.hugo)
                    geneTargets[genename.text] = locus.text
        return geneTargets
    
    def fetch_gene_targets(self, gene: str) -> list:
        return

    def fetch_targets(self, dg: str, mode: str = "drug"):
        if(mode=="drug"):
            return list((self.fetch_drug_targets(dg)).keys())
        elif(mode=="gene"):
            return list((self.fetch_gene_targets(dg)).keys())
    
    def fetch_drugs(self):
        drugs = []
        # Iterate through individual drugs from drugbank
        drugBank, prefix = self.main_tree.getroot(), self.dbprefix
        for drugelement in drugBank:
            # Get the drug name and check if it is the desired drug ('drugTarget')
            namecheck = (drugelement.find(prefix+"name")).text.lower().strip()
            drugs.append(namecheck)
        return drugs