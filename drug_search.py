#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 2025

@author: jds40
"""

import os
import numpy as np
import pandas as pd
from lxml import etree
from tqdm import tqdm
from urllib import request
import re

from typing import Optional, Union

CLEANED_DATA_DIR: str = os.path.join("Data", "Laurence-Data")
DEFAULT_HUGO_FILE: str = os.path.join(CLEANED_DATA_DIR, "hgnc_complete_set.tsv")

DRUG_TAG_PREFIX: str = "{http://www.drugbank.ca}"

def get_targets_all(data: Optional[Union[list, tuple]] = None,
                hgncdata: Optional[pd.DataFrame] = None,
                gdscoverview: Union[pd.DataFrame, str] = os.path.join("Data", "Results", "GDSCdrugs.tsv")) -> pd.DataFrame:
    # Load in DrugBank and GDSC data
    if(data is None):
        db, g1, g2 = get_data()
    # Load in HGNC data
    if(hgncdata is None):
        hgncdata = pd.read_table(DEFAULT_HUGO_FILE, low_memory=False).fillna('')
    ## Load in GDSC overview data
    # Check GDSC overview file exists
    if(type(gdscoverview)==str):
        # Try to locate and/or download GDSC overview file
        if(os.path.exists(gdscoverview)==False):
            gdscoverview = os.path.join("Data", "Results", "GDSCdrugs.tsv")
            if(os.path.exists(gdscoverview)==False):
                print(f"GDSC Overview file not found. Attempting to download... to 'Data/Results/GDSCdrugs.tsv'")
                try:
                    request.urlretrieve("https://www.cancerrxgene.org/api/compounds?list=all&sEcho=1&iColumns=7&sColumns=&iDisplayStart=0&iDisplayLength=25&mDataProp_0=0&mDataProp_1=1&mDataProp_2=2&mDataProp_3=3&mDataProp_4=4&mDataProp_5=5&mDataProp_6=6&sSearch=&bRegex=false&sSearch_0=&bRegex_0=false&bSearchable_0=true&sSearch_1=&bRegex_1=false&bSearchable_1=true&sSearch_2=&bRegex_2=false&bSearchable_2=true&sSearch_3=&bRegex_3=false&bSearchable_3=true&sSearch_4=&bRegex_4=false&bSearchable_4=true&sSearch_5=&bRegex_5=false&bSearchable_5=true&sSearch_6=&bRegex_6=false&bSearchable_6=true&iSortCol_0=0&sSortDir_0=asc&iSortingCols=1&bSortable_0=true&bSortable_1=true&bSortable_2=true&bSortable_3=true&bSortable_4=true&bSortable_5=true&bSortable_6=true&export=tsv",
                    gdscoverview)
                except:
                    pass
        # Try to read in GDSC overview file
        if(os.path.exists(gdscoverview)):
            gdscoverview = pd.read_csv(gdscoverview, sep = "\t")
        else:
            print(f"Failed to load GDSC overview table; cannot get drugs to check")
    output = {}
    for drug in tqdm(gdscoverview[" Name"].unique(), desc = "Fetching available drug targets"):
        output[drug] = get_targets(drug, (db, g1, g2), hgncdata, gdscoverview)
    # Transpose list to get database as columns and drugs as rows
    df = pd.DataFrame(output).T
    # Integrate manual PubChem identifications
    if(os.path.exists(os.path.join("Data", "Results", "manual_drug_translations.tsv"))):
        sdf = pd.read_csv(os.path.join("Data", "Results", "manual_drug_translations.tsv"), sep = "\t")
        for oname, pcid in zip(sdf["GDSC Drug Name"].values, sdf["PubChem"].values):
            # Ensure not dealing with a NaN value
            if(pcid==pcid):
                df.loc[df.index == oname, "PubChem"] = str(pcid)
    # Amend the PubChem information to replace empty lists and lists of nan/none to be NaN values
    newcol = df["PubChem"].tolist()
    for i in range(len(newcol)):
        if(i<50):
            print(f"Pre-processing: {newcol[i]}")
        newcol[i] = str(newcol[i])
        if("[" in newcol[i]):
            # Split using commas and spaces as delimiters
            newcol[i] = re.split(",| ", newcol[i])
            # Remove any opening/closing brackets
            for j in range(len(newcol[i])):
                newcol[i][j] = newcol[i][j].replace("[","").replace("]", "")
        elif(len(newcol[i])>0 and "nan" not in newcol[i].lower() and "none" not in newcol[i].lower()):
            newcol[i] = [newcol[i],]
        else:
            newcol[i] = []
            continue
        if(i<50):
            print(f"Pre-nan removal: {newcol[i]}")
        for j in range(len(newcol[i]))[::-1]:
            # Remove any apostrophes/speech marks
            newcol[i][j] = str(newcol[i][j]).replace("\'","").replace("\"", "")
            # Remove NaN entries
            entry = newcol[i][j]
            if(entry.lower() in ["nan", "none",""]):
                newcol[i].pop(j)
    df["PubChem"] = newcol

    # Replace empty lists and empty dictionaries as NaN values
    df.mask(df.map(str).isin(["[]", "{}"]), inplace = True)
    #df=df.where(df.astype(bool),np.nan, inplace = False)
    #df.replace({"DrugBank": {{}: np.nan}, "GDSC": {[]: np.nan}}, inplace=True)
    return df

def get_targets(drug_selected: str, data: Optional[Union[list, tuple]] = None,
                hgncdata: Optional[pd.DataFrame] = None,
                gdscoverview: Union[pd.DataFrame, str] = os.path.join("Data", "Results", "GDSCdrugs.tsv")) -> dict:
    """
    Function to retrieve official, known targets for a given drug
    
    :param drug_selected: Description
    :type drug_selected: str
    :param data: Description
    :type data: Optional[Union[list, tuple]]
    :param hgncdata: Description
    :type hgncdata: Optional[pd.DataFrame]
    :param gdscoverview: Description
    :type gdscoverview: Union[pd.DataFrame, str]
    :return: Description
    :rtype: dict
    """
    # Load in the data with the get_data function; this loads GDSC1 and GDSC2 as pandas dataframes, and DrugBank as an lxml ElementTree object
    if(data is None):
        db, g1, g2 = get_data()
    else:
        db, g1, g2 = data
    # If there are any load-in failures, stop the program
    if(db is None or g1 is None or g2 is None):
        return

    # Select drug - currently this is just the first value from GDSC1 but in theory it could be any drug name from GDSC1/2
    #drug_selected = g1["DRUG_NAME"].values[0]
    
    # Get the DrugBank tree as a root; some information about this tree:
    #   The root tag is '{http://www.drugbank.ca}drugbank', while each subitem has the '{http://www.drugbank.ca}drug' tag
    root = db.getroot()

    # Update gene names in gdsc1/2 databases using HUGO if given
    if(hgncdata is None):
        hgncdata = pd.read_table(DEFAULT_HUGO_FILE, low_memory=False).fillna('')
    
    output: dict = {}
    
    # Find the gene names and loci targeted by the drug according to DrugBank
    output["DrugBank"] = find_targets_drugbank(drug_selected, root, hgncdata)

    # Check GDSC overview file exists
    if(type(gdscoverview)==str):
        # Try to locate and/or download GDSC overview file
        if(os.path.exists(gdscoverview)==False):
            gdscoverview = os.path.join("Data", "Results", "GDSCdrugs.tsv")
            if(os.path.exists(gdscoverview)==False):
                print(f"GDSC Overview file not found. Attempting to download... to 'Data/Results/GDSCdrugs.tsv'")
                try:
                    request.urlretrieve("https://www.cancerrxgene.org/api/compounds?list=all&sEcho=1&iColumns=7&sColumns=&iDisplayStart=0&iDisplayLength=25&mDataProp_0=0&mDataProp_1=1&mDataProp_2=2&mDataProp_3=3&mDataProp_4=4&mDataProp_5=5&mDataProp_6=6&sSearch=&bRegex=false&sSearch_0=&bRegex_0=false&bSearchable_0=true&sSearch_1=&bRegex_1=false&bSearchable_1=true&sSearch_2=&bRegex_2=false&bSearchable_2=true&sSearch_3=&bRegex_3=false&bSearchable_3=true&sSearch_4=&bRegex_4=false&bSearchable_4=true&sSearch_5=&bRegex_5=false&bSearchable_5=true&sSearch_6=&bRegex_6=false&bSearchable_6=true&iSortCol_0=0&sSortDir_0=asc&iSortingCols=1&bSortable_0=true&bSortable_1=true&bSortable_2=true&bSortable_3=true&bSortable_4=true&bSortable_5=true&bSortable_6=true&export=tsv",
                    gdscoverview)
                except:
                    pass
        # Try to read in GDSC overview file
        if(os.path.exists(gdscoverview)):
            gdscoverview = pd.read_csv(gdscoverview, sep = "\t")
        else:
            print(f"Failed to load GDSC overview table; target results will be lacking")
    
    # Load targets and PubChem from GDSC overview file
    if(type(gdscoverview)==pd.DataFrame):
        #gdscoverview[" Targets"].replace("", np.nan, inplace=True)
        gdscoverview.replace({" Targets": {"": np.nan}}, inplace=True)
        gdscoverview.dropna(subset=[" Targets"], inplace=True)
        if(drug_selected in gdscoverview[" Name"].values):
            output["GDSC"] = gdscoverview.loc[gdscoverview[" Name"]==drug_selected][" Targets"].values
            output["PubChem"] = gdscoverview.loc[gdscoverview[" Name"]==drug_selected][" PubCHEM"].values
        else:
            output["GDSC"] = []
            output["PubChem"] = []
    
    return output

def find_targets_drugbank(drugTarget: str, drugBank: etree.ElementTree, hgncdata: pd.DataFrame) -> dict:

    # Set up dictionary which will be populated by gene targets
    gene_targets = {}

    # Iterate through individual drugs from drugbank
    for drugelement in drugBank:
        # Get the drug name and check if it is the desired drug ('drugTarget')
        namecheck = drugelement.find(DRUG_TAG_PREFIX+"name")
        if(drugTarget.lower().strip()==namecheck.text.lower().strip()):
            # Get the gene targets of this drug
            targets = drugelement.find(DRUG_TAG_PREFIX+"targets")
            # Iterate through the gene targets, find each one's name and locus, and add this to the initial dictionary
            for target in targets.findall(DRUG_TAG_PREFIX+"target"):
                polypeptide = target.find(DRUG_TAG_PREFIX+"polypeptide")
                # If there is no polypeptide found, add this target as an exception
                if(polypeptide is None):
                    gene_targets.update({f"{target} (Irregular target)": None})
                    continue
                genename, locus, cl = polypeptide.find(DRUG_TAG_PREFIX+"gene-name"), polypeptide.find(DRUG_TAG_PREFIX+"locus"), polypeptide.find(DRUG_TAG_PREFIX+"chromosome-location")
                # Update genename if necessary
                genename = update_hgnc_single(genename, hgncdata)
                gene_targets.update({genename.text: locus.text})
    
    return gene_targets

def get_data() -> tuple[Optional[etree.ElementTree], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    # Set up initial variables (important in case any files aren't found)
    db, g1, g2 = None, None, None
    # Make the relevant directories if they do not exist
    if(os.path.exists("Data")==False):
        os.mkdir("Data")
        os.mkdir(os.path.join("Data","Raw Data"))
    # Iterate through raw data files. Import GDSC1 and GDSC2 as pandas Dataframes, import DrugBank as an lxml.etree ElementTree
    for filename in os.listdir(os.path.join("Data", "Raw Data")):
        if("gdsc1" in filename.lower() and filename.lower()[-5:]==".xlsx"):
            g1 = pd.read_excel(os.path.join("Data", "Raw Data", filename), engine = "openpyxl")
            print(f"Imported GDSC1 from {filename}")
        elif("gdsc2" in filename.lower() and filename.lower()[-5:]==".xlsx"):
            g2 = pd.read_excel(os.path.join("Data", "Raw Data", filename), engine = "openpyxl")
            print(f"Imported GDSC2 from {filename}")
        elif(filename.lower()=="full database.xml"):
            with open(os.path.join("Data", "Raw Data", filename), "r") as f:
                db = etree.parse(f)
            print(f"Imported DrugBank from {filename}")
    # If one or more of the data files weren't found, report this as an error
    if(db is None or g1 is None or g2 is None):
        errors = ""
        if(db is None):
            errors += "DrugBank, "
        if(g1 is None):
            errors += "GDSC1, "
        if(g2 is None):
            errors += "GDSC2, "
        errors = errors[:-2]
        print(f"Error: cannot find file for {errors}")
    return db, g1, g2

def make_dir(directory_to_make: str) -> None:
    path = ""
    for part in directory_to_make.split(os.sep):
        path = os.path.join(path, part)
        if(os.path.exists(path)==False):
            os.mkdir(path)
    return

def update_hgnc(df: pd.DataFrame, hgncdata: pd.DataFrame, column: str = "symbol") -> pd.DataFrame:
    """
    Normalise archaic names using HGNC standard

    Args:
        df (pd.DataFrame): DataFrame with names to replace using HGNC
        hgncdata (pd.DataFrame): HGNC data

    Returns:
        pd.DataFrame: Version of 'df' with updated gene names
    """
    bad_names = set(df[column]) & (set(df[column]) ^ set(hgncdata.index))

    for g in tqdm(bad_names, desc = "Replacing gene names using HUGO standardisation"):
        g2 = hgncdata[hgncdata['prev_symbol'].str.contains(g, na=False)].reset_index()['symbol']
        if len(g2) == 0 or (g2[0] not in hgncdata.index):
            pass
            #print(f'STRING Gene name {g} not found in HUGO - ignoring it')
        else:
            #print(f'STRING old gene name {g} replaced by new name {g2[0]}')
            df.replace(g,g2[0], inplace=True)
    return df

def update_hgnc_single(gn: str, hgncdata: pd.DataFrame) -> str:
    """
    Normalise individual archaic name using HGNC standard

    Args:
        gn (str): Single gene name to replace using HGNC
        hgncdata (pd.DataFrame): HGNC data

    Returns:
        pd.DataFrame: Version of 'df' with updated gene names
    """
    if(gn in hgncdata["prev_symbol"]):
        return hgncdata.loc[hgncdata["prev_symbol"] == gn]["symbol"].values[0]
    else:
        return gn