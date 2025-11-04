#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 08:55:06 2024

@author: lp217
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

import smtplib, ssl
from email.mime.text import MIMEText
import email.utils


def SendMail(destination, message):
    mymail = "pearlaicode24@gmail.com"
    password = "iujr uusm zqdv yryu"
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


def ChunkDrugGene(it,dl,deps,drug1,drug2):
    result = pd.DataFrame(columns=['symbol']+dl)
    result.symbol = list(deps.columns[1:])
    result = result.fillna(0.0)
    result.set_index('symbol', inplace=True)

    # loop through all drugs, calculating r for all genes


    for d in dl:
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
                
                x = np.array(dpdat1[gn])
                # X = x[:,np.newaxis]
                y = dpdat1['pKi']
                                
                pr1,pp1 = pearsonr(x,y)
                # print(f'{d} - {gn} : r = {pr1:.3f}, p = {pp1:.3g}')

            if len(dep2) > 0:
                
                x = np.array(dpdat2[gn])
                # X = x[:,np.newaxis]
                y = dpdat2['pKi']
                                
                pr2,pp2 = pearsonr(x,y)
                # print(f'{d} - {gn} : r = {pr2:.3f}, p = {pp2:.3g}')
                
    # select which value to use

            result.at[gn,d] = max(pr1,pr2,key=abs)
            if (i % 1000) == 0:
                print(f'Thread {it} done {i} genes',flush=True)
            
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

    f = default_dep_dat_file = 'CRISPRGeneDependency.csv'
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
    f = default_HUGO_file = 'hgnc_complete_set.tsv'
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
    
    
    
    f = default_cell_info_file = 'Model.csv'
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
    
    f = default_drug1_file = 'GDSC1_drug_results_target_cleaned7.tsv'
    # if spec:
    #     f = input(f'GDSC1 Gene file [{default_drug1_file}] : ')
    #     if f.strip() == '':
    #         f = default_drug1_file
    print(f'Loading {f}')
    drug1 = pd.read_table(f, low_memory=False).fillna('')
    drug1.DRUG_NAME = drug1.DRUG_NAME.apply(lambda x:x.upper())
    
    
    f = default_drug2_file = 'GDSC2_drug_results_target_cleaned7.tsv'
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
    
    SendMail('laurencepearl@btinternet.com',mess)
    
    nested_dfs = pool.starmap_async(ChunkDrugGene,
            [(i,batch_dlist[i],deps,drug1,drug2) 
             for i in range(usable_CPUs)]).get()
    
    print(f'All by All took {(timer()-t_base)/60.0:.4} min')
    
    allbyall = pd.concat(nested_dfs,axis=1)   
    
    print('Writing Drugs x Genes file)')
    allbyall.to_csv('AllDrugsByAllGenes.tsv', sep='\t', index=True, header=True)
    print('Writing Genes x Drugs file)')
    allbyall.rename({'symbol': 'drug'}, axis='columns').set_index('drug').T.to_csv('AllGenesByAllDrugs.tsv', sep='\t', index=True, header=True)
    
    
    
    mess = f'{len(dlist)} Drugs x {len(deps.columns[1:])} Genes correlation calculation complete'
    
    SendMail('laurencepearl@btinternet.com',mess)
            

            

    
    
    
    
    
    
    
    
    
    
    
    
        
        

