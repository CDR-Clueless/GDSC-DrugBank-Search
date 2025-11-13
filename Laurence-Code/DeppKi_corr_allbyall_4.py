#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 08:55:06 2024

@author: lp217
@coauthor: jds40
"""

import pandas as pd
import numpy as np
# from matplotlib.patches import Rectangle, Ellipse
# import matplotlib.patches as mpatch
from scipy.stats import pearsonr
# from scipy.signal import argrelextrema, find_peaks
from timeit import default_timer as timer
import multiprocessing as mp
# from igraph import Graph,plot
import json

import smtplib, ssl
from email.mime.text import MIMEText
import email.utils

from copy import deepcopy
import os

CLEANED_DATA_DIR: str = os.path.join("Data", "Laurence-Data")
DEFAULT_DEP_DAT_FILE: str = os.path.join(CLEANED_DATA_DIR,"CRISPRGeneDependency.csv")
DEFAULT_HUGO_FILE: str = os.path.join(CLEANED_DATA_DIR, "hgnc_complete_set.tsv")
DEFAULT_CELL_INFO_FILE: str = os.path.join(CLEANED_DATA_DIR, "Model.csv")
DEFAULT_DRUG1_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC1_drug_results_target_cleaned7.tsv')
DEFAULT_DRUG2_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC2_drug_results_target_cleaned7.tsv')
with open(os.path.join("Local", "passwords.json"), "r") as f:
    DESTINATION_EMAIL = json.load(f)["destination-email"]

def SendMail(destination, message):
    with open(os.path.join("Local", "passwords.json"), "r") as f:
        login = json.load(f)
    mymail = login["pearlaiemail"]["email"]
    password = login["pearlaiemail"]["password"]
    receiver = destination
    message = MIMEText(message, "plain")
    message["Subject"] = "Message from PearlAI"
    message["From"] = email.utils.formataddr(('PearlAI', mymail))
    
    port = 465
    sslcontext = ssl.create_default_context()
    connection = smtplib.SMTP_SSL(
        "smtp.gmail.com",
        port,
        context=sslcontext
    )
    
    connection.login(mymail, password)
    connection.sendmail(mymail, receiver, message.as_string())
    connection.quit()
    
    print("Email sent")
    
def split_list(l: list, parts: int) -> list:

    n = min(parts, max(len(l),1))
    k, m = divmod(len(l), n)
    return [l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def ChunkDrugGene(it: int, dl: set, deps: pd.DataFrame, drug1: pd.DataFrame, drug2: pd.DataFrame):
    """
    
    Function for calculating Drug-gene correlations for a chunk of drugs (necessary for parallel processing)

    Args:
        it (int): (Simplified) ID for the thread working on this chunk
        dl (set): Set of drugs in this chunk to be tested
        deps (pd.DataFrame): DataFrame of CRISPR gene dependency data (DepMap)
        drug1 (pd.DataFrame): DataFrame of GDSC1 data
        drug2 (pd.DataFrame): DataFrame of GDSC2 data
    """

    result = pd.DataFrame(columns=['symbol']+dl)
    result.symbol = list(deps.columns[1:])
    result = result.fillna(0.0)
    result.set_index('symbol', inplace=True)

    # loop through all drugs, calculating r for all genes
    for d in dl:
        # Load the calculation for this data if it has already been calculated
        starfiledir = os.path.join(CLEANED_DATA_DIR, "temp_starmap_store", f"starmapcorrelations-{d}")
        if(os.path.exists(starfiledir)):
           dresult = pd.read_csv(os.path.join(CLEANED_DATA_DIR, "temp_starmap_store", f""), sep = "\t", index = False, lineterminator="\n")
           result[:, d] = deepcopy(dresult)
           print(f"Thread {it} found and loaded correlations for {d}", flush = True)
           continue

        # If there is no file, calculate the correlations for this drug
        print(f'Thread {it} Calculating correlations for {d}',flush=True)

        c1 = drug1[drug1.DRUG_NAME == d]['ModelID']
        c2 = drug2[drug2.DRUG_NAME == d]['ModelID']
                
        for i,gn in enumerate(deps.columns[1:]):
            
            # get deps
            dep1 = deps[deps.index.isin(c1)][gn].reset_index()
            dep1_names = list(dep1.ModelID)
            dep2 = deps[deps.index.isin(c2)][gn].reset_index()
            dep2_names = list(dep2.ModelID)
            
            # get pKis
            if len(dep1) > 0:
                # Get first available pKi value for relevant drug with acceptable ModelID values
                pKi1 = drug1[(drug1.DRUG_NAME == d) & 
                             (drug1.ModelID.isin(dep1_names))].drop_duplicates \
                            (subset=["ModelID"], keep='first')[['pKi','ModelID']]
                # merge data on common ModelID
                dpdat1 = dep1.merge(pKi1)
                
            if len(dep2) > 0:
                pKi2 = drug2[(drug2.DRUG_NAME == d) & 
                             (drug2.ModelID.isin(dep2_names))].drop_duplicates \
                            (subset=["ModelID"], keep='first')[['pKi','ModelID']]
                # merge data on common ModelID
                dpdat2 = dep2.merge(pKi2)
                
            pr1,pp1,pr2,pp2 = 0,1,0,1

            if len(dep1) > 0:
                
                # Correlation coefficient between current gene expression (0 or 1 as it's binary k/o) and pKi value
                x = np.array(dpdat1[gn])
                y = dpdat1['pKi']
                                
                pr1,pp1 = pearsonr(x,y)
                # print(f'{d} - {gn} : r = {pr1:.3f}, p = {pp1:.3g}')

            if len(dep2) > 0:
                
                x = np.array(dpdat2[gn])
                # X = x[:,np.newaxis]
                y = dpdat2['pKi']
                                
                pr2,pp2 = pearsonr(x,y)
                # print(f'{d} - {gn} : r = {pr2:.3f}, p = {pp2:.3g}')
                
            # select which value to use; pr1 or pr2

            result.at[gn,d] = max(pr1,pr2,key=abs)
            if (i % 1000) == 0:
                print(f'Thread {it} done {i} genes',flush=True)

        # Save result for this valud of 'd', in case the program is interrupted
        result[d].to_csv(starfiledir, sep = "\t", index = False, lineterminator="\n")
            
    return(result)
            
        

def UpdateGeneName(g):
    if g in hgnc.index:
        return g
    elif g in list(hgnc.prev_symbol):
        return hgnc[hgnc.prev_symbol.str.contains(f'^{g}$',regex=True)].index[0]
    else:
        return ''

# =============================================================================
# ███╗   ███╗     █████╗     ██╗    ███╗   ██╗
# ████╗ ████║    ██╔══██╗    ██║    ████╗  ██║
# ██╔████╔██║    ███████║    ██║    ██╔██╗ ██║
# ██║╚██╔╝██║    ██╔══██║    ██║    ██║╚██╗██║
# ██║ ╚═╝ ██║    ██║  ██║    ██║    ██║ ╚████║
# ╚═╝     ╚═╝    ╚═╝  ╚═╝    ╚═╝    ╚═╝  ╚═══╝
#
# =============================================================================
if __name__ == '__main__':
    print("Number of processors: ", mp.cpu_count())

    f = DEFAULT_DEP_DAT_FILE
    # f = default_dep_dat_file = 'CRISPRGeneEffect.csv'
    
    # if spec:
    #     f = input(f'Gene dependency file [{default_dep_dat_file}] : ')
    #     if f.strip() == '':
    #         f = default_dep_dat_file
    print(f'Loading {f}')
    
    deps = pd.read_csv(f).fillna(0.0)
    
    default_dep_dat_file = f
    
    deps_loaded = True
    
    # make cell line index to dependency data
    
    deps.rename(columns = {'Unnamed: 0':'ModelID'},inplace=True)
    deps.set_index('ModelID', inplace=True)
    
    # edit header names to conform to remove extraneous data
    
    gg = dict(zip(list(deps.columns), [i.split()[0]
              for i in list(deps.columns)]))
    deps.rename(columns=gg, inplace=True)
    # print(deps)
    
    # load HUGO chromosomal location data
    f = DEFAULT_HUGO_FILE
    # if spec:
    #     f = input(f'HUGO Gene file [{default_HUGO_file}] : ')
    #     if f.strip() == '':
    #         f = default_HUGO_file
    print(f'Loading {f}')
    hgnc = pd.read_table(f, low_memory=False).fillna('')
    hgnc = hgnc[['symbol', 'ensembl_gene_id',
                 'prev_symbol', 'location', 'location_sortable']]
    hgnc.set_index('symbol', inplace=True)
    hgnc_loaded = True
    # print(hgnc)
    # correct legacy deps gene symbols to HUGO
    
    bad_names = set(deps.columns) & (set(hgnc.index) ^ set(deps.columns))
    print(f'Updating {len(bad_names)} archaic gene names in dependency data')
    for g in bad_names:
        g2 = hgnc[hgnc['prev_symbol'].str.contains(g)].reset_index()['symbol']
        if len(g2) == 0 or (g2[0] not in hgnc.index):
            # print(f'DepMap Gene name {g} not found in HUGO - ignoring it')
            continue
        else:
            # print(f'DepMap old gene name {g} replaced by new name {g2[0]}')
            deps.rename(columns={g: g2[0]}, inplace=True)
    
    
    
    f = DEFAULT_CELL_INFO_FILE
    # if spec:
    #     f = input(
    #         f'Cell sample information file [{default_cell_info_file}] : ')
    #     if f.strip() == '':
    #         f = default_cell_info_file
    print(f'Loading {f}')
    
    info = pd.read_csv(f, low_memory=False).fillna('')
    info_loaded = True
    info['OncotreeLineage'] = [x.upper() for x in info['OncotreeLineage']]
    info.OncotreePrimaryDisease = info.OncotreePrimaryDisease.str.replace(' ','_')
    cancer_types = set(info['OncotreeLineage'])
    
    
    # load drug data
    
    f = DEFAULT_DRUG1_FILE
    # if spec:
    #     f = input(f'GDSC1 Gene file [{default_drug1_file}] : ')
    #     if f.strip() == '':
    #         f = default_drug1_file
    print(f'Loading {f}')
    drug1 = pd.read_table(f, low_memory=False).fillna('')
    drug1.DRUG_NAME = drug1.DRUG_NAME.apply(lambda x:x.upper())
    
    
    f = DEFAULT_DRUG2_FILE
    # if spec:
    #     f = input(f'GDSC2 Gene file [{default_drug2_file}] : ')
    #     if f.strip() == '':
    #         f = default_drug2_file
    print(f'Loading {f}')
    drug2 = pd.read_table(f, low_memory=False).fillna('')
    drug2.DRUG_NAME = drug2.DRUG_NAME.apply(lambda x:x.upper())
    
    # initilise all by all data
    
    dlist = sorted(set(drug1.DRUG_NAME) | set(drug2.DRUG_NAME))
    
    allbyall = pd.DataFrame(columns=['symbol']+dlist)
    allbyall.symbol = list(deps.columns[1:])
    allbyall = allbyall.fillna(0.0)
    allbyall.set_index('symbol', inplace=True)
    
    # set up parallel loop
    
    t_base = timer()
    
    usable_CPUs = mp.cpu_count()-2
    pool = mp.Pool(usable_CPUs)
       
    # batch up  calcualtions 
    
    batch_dlist = split_list(dlist,usable_CPUs)    
    nested_dfs = []
    
    print(f'splitting data into {len(batch_dlist)} batches ...')
    
    mess = f'Starting {len(dlist)} Drugs x {len(deps.columns[1:])} Genes correlation calculation'
    
    try:
        SendMail(DESTINATION_EMAIL,mess)
    except:
        print(f"Failed to send e-mail 1:\n{mess}")
    
    # Set up directory to store temporary calculations from parallel functions
    if(os.path.exists(os.path.join(CLEANED_DATA_DIR, "temp_starmap_store"))==False):
        os.mkdir(os.path.join(CLEANED_DATA_DIR, "temp_starmap_store"))

    nested_dfs = pool.starmap_async(ChunkDrugGene,
            [(i,batch_dlist[i],deps,drug1,drug2) 
             for i in range(usable_CPUs)]).get()
    
    print(f'All by All took {(timer()-t_base)/60.0:.4} min')
    
    allbyall = pd.concat(nested_dfs,axis=1)   
    
    print('Writing Drugs x Genes file)')
    allbyall.to_csv(os.path.join(CLEANED_DATA_DIR, 'AllDrugsByAllGenes.tsv'), sep='\t', index=True, header=True)
    print('Writing Genes x Drugs file)')
    allbyall.rename({'symbol': 'drug'}, axis='columns').set_index('drug').T.to_csv(os.path.join(CLEANED_DATA_DIR, 'AllGenesByAllDrugs.tsv'), sep='\t', index=True, header=True)
    
    # Delete the temporary data store
    for filename in os.listdir(os.path.join(CLEANED_DATA_DIR, "temp_starmap_store")):
        os.remove(os.path.join(CLEANED_DATA_DIR, "temp_starmap_store", filename))
    os.rmdir(os.path.join(CLEANED_DATA_DIR, "temp_starmap_store"))
    
    mess = f'{len(dlist)} Drugs x {len(deps.columns[1:])} Genes correlation calculation complete'
    
    try:
        SendMail(DESTINATION_EMAIL,mess)
    except:
        print(f"Failedd to send e-mail 2:\n{mess}")
            

            

    
    
    
    
    
    
    
    
    
    
    
    
        
        

