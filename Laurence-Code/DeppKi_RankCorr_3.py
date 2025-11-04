#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 08:55:06 2024

@author: lp217
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from math import ceil,log10
from collections import OrderedDict, Counter


# from matplotlib.patches import Rectangle, Ellipse
# import matplotlib.patches as mpatch
# from scipy.signal import argrelextrema, find_peaks
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


def UpdateGeneName(g):
    if g in hgnc.index:
        return g
    elif g in list(hgnc.prev_symbol):
        return hgnc[hgnc.prev_symbol.str.contains(f'^{g}$',regex=True)].index[0]
    else:
        return ''

def sort_corrs(d):
    return corrs[['symbol',d]].sort_values(
        by=d,ascending =False).reset_index().drop('index',axis=1)

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

drug_names = set(drug1.DRUG_NAME) | set(drug2.DRUG_NAME)

# load AllByAll data

corrs = pd.read_table('AllDrugsByAllGenes.tsv').fillna(0.0)


# regval = {}
results = pd.DataFrame(columns=['DRUG','TARGET','RANK','TOP10'])

# make drug name list

dlist = list(corrs.columns[1:])

print(f'{len(dlist)} drugs to analyse')

# loop through drug list looking for targets

for d in dlist:
    
# get correlations for this drug as sorted list

    dc = sort_corrs(d)
    
# get target gene name(s)
    
    t1 = drug1[drug1.DRUG_NAME == d]['PUTATIVE_TARGET'].unique()
    t2 = drug2[drug2.DRUG_NAME == d]['PUTATIVE_TARGET'].unique()
    if len(t1) > 0:
        t = [x.upper().strip() for x in t1[0].split(',')]
    elif len(t2) > 0:
        t = [x.upper().strip() for x in t2[0].split(',')]

    # t = list(drug1[drug1.DRUG_NAME == d]['PUTATIVE_TARGET'].unique())[0].split(',')
    # t = [x.upper().strip() for x in t]
    
    for g in t: # extract target gene names from list
        gn = UpdateGeneName(g)    
        if gn == '':
            print(f"Can't match {g} to a gene name for drug {d} - skipping")
            continue
    
# get index of match to target
        

        i = dc.index[dc.symbol == gn].tolist()
        if len(i) == 0:
            print(f'No correlation data for Gene {gn} with Drug {d} - skipping ..')
            continue
        i = i[0]
        print(f'Drug {d} - target {gn} - rank {i}')
    
        top10 = dc[dc[d] > 0.0]['symbol'].to_list()[:10]
        results.loc[len(results)] = [d,gn,i,top10]
        
results.to_csv('TargetRanking.tsv', sep='\t', index=False, header=True)

# plot histogram of target ranking. For multiple targets take highest ranking of group

fig,ax = plt.subplots()

rc = []

for d in set(results.DRUG):
    rc.append(ceil(log10(results[results.DRUG == d].sort_values('RANK')['RANK'].min()+1)))
plt.tick_params(axis='both', labelsize=10)
plt.xticks([0,1,2,3,4,5],['1','2-10','11-100','101-1000','1001-10000','10001-'])
plt.hist(rc,bins=np.arange(7)-0.5,align='mid',rwidth=0.5)
plt.xlabel('Target rank position',fontsize=15)
plt.ylabel('N',rotation='horizontal',fontsize=15)
plt.title('Target Dependency/Drug Efficacy Correlation',fontsize=15)
plt.show()

SendMail('laurencepearl@btinternet.com','Drug/Gene Ranking complete')


# read in human genome interactions from STRING

stinfo = pd.read_table('9606.protein.info.v12.0.txt',usecols=['#string_protein_id','preferred_name']).fillna('')

stinfo.rename(columns = {'#string_protein_id':'ID','preferred_name':'symbol'},inplace=True)

# normalise any archaic names

bad_names = set(stinfo.symbol) & (set(stinfo.symbol) ^ set(hgnc.index))

for g in bad_names:
    g2 = hgnc[hgnc['prev_symbol'].str.contains(g)].reset_index()['symbol']
    if len(g2) == 0 or (g2[0] not in hgnc.index):
        print(f'STRING Gene name {g} not found in HUGO - ignoring it')
        continue
    else:
        print(f'DepMap old gene name {g} replaced by new name {g2[0]}')
        stinfo.replace(g,g2[0], inplace=True)


stdict = dict(zip(stinfo.ID,stinfo.symbol))

stlink = pd.read_csv('9606.protein.links.v12.0.txt',sep=' ')

# filter links to combined_score > 0.8

stlink.combined_score = stlink.combined_score.astype(int)
stlink[stlink.combined_score.gt(800)]

stlink.protein1 = stlink.protein1.apply(lambda x: stdict[x])
stlink.protein2 = stlink.protein2.apply(lambda x: stdict[x])






# make source graph

print('Building human link graph ...')
g = ig.Graph.TupleList(stlink.itertuples(index=False), directed=True, weights=False, edge_attrs="combined_score")
print('Done !!')

# sort data on target rank

results = results.sort_values('RANK')

for i in range(len(results)):
    targ = results.iloc[i].TARGET
    top10 = results.iloc[i].TOP10
    d = results.iloc[i].DRUG

# loop through drugs analysing network link from target to top10 corr hits
    print(f'Drug : {d} Target : {targ}')

    
# find shortest path from target to each of the top10 and accumulate vertices

    cliq = []
    
    for t in top10:
        sp = g.get_shortest_paths(targ, to=t, weights=g.es["combined_score"], output="vpath")[0]
        cliq += sp
    cliq = list(set(cliq))

# find encompassing subgraph

    subg = g.subgraph(cliq)
    
# make unidirectional

    ssg = subg.as_undirected().simplify()
    
# find communities

    com = ssg.community_edge_betweenness().as_clustering()
    num_com = len(com)
    # palette = ig.ClusterColoringPalette(n=num_com)
    palette = ig.AdvancedGradientPalette(["pink", 'palegoldenrod',"palegreen", "powderblue"], n=num_com)
    for i, community in enumerate(com):
        ssg.vs[community]["color"] = i
        community_edges = ssg.es.select(_within=community)
        community_edges["color"] = i
    
# transfer edge weights from bidirectional graph

    for i,e in enumerate(ssg.es):
        ss = e.source
        tt = e.target
        for f in subg.es:
            if f.source == ss and f.target == tt:
                ssg.es[i]['combined_score'] = f['combined_score']
        
        
# get edge values

    pvals = [p for p in ssg.es['combined_score']]
    pmax = max(pvals)
    pmin = min(pvals)
    prange = pmax-pmin
    
    fig,ax = plt.subplots(figsize=(10,10))
    ax.set_title(f'Drug : {d} - Target : {targ}', fontsize=16)
    
    # ssg.vs['label'] = ssg.vs['name']
    ssg.vs['label'] = [x if x not in top10 else f'{top10.index(x)}\n{x}' for x in ssg.vs['name']]
    lay = ssg.layout('kk')
    style = {}
    style['layout'] = lay
    # style['autocurve'] = True
    # style['bbox'] = (1024,1024)
    # style['margin'] = 50
    style['vertex_size'] = [500 if n == targ else 350 if n in top10 else 250 for n in ssg.vs['name']]
    style['vertex_color'] = ['pink' if n == targ else 'moccasin' if n in top10 else 'white' for n in ssg.vs['name']]
    # style['edge_arrow_size'] = 0.75
    # #style['edge_width'] = [min([p,10]) for p in gg.es['p']]
    style['edge_width'] = [10*(p-pmin)/prange+2.0 for p in pvals]
    # style['edge_arrow_size'] = [max(0.5,2.2*(p-pmin)/prange+0.1) for p in ssg.es['combined_score']]

    ig.plot(com,palette=palette,target=ax,**style)
    # print(targ+'_net2.pdf created' )
    plt.show()




# dlist = sorted(list(set(results.DRUG)))

# for d in dlist:
#     d1 = results[results.DRUG == d]
#     for d2 in d1.index:
        
# # get target and top10 scoring
#         targ = d1.loc[d2].TARGET
#         top10 = d1.loc[d2].TOP10
        
#         print(f'Drug : {d} Target : {targ}')

    
# # find shortest path from target to each of the top10 and accumulate vertices

#         cliq = []
        
#         for t in top10:
#             sp = g.get_shortest_paths(targ, to=t, weights=g.es["combined_score"], output="vpath")[0]
#             cliq += sp
#         cliq = list(set(cliq))

# # find encompassing subgraph

#         subg = g.subgraph(cliq)
        
# # make unidirectional

#         ssg = subg.as_undirected().simplify()
        
# # find communities

#         com = ssg.community_edge_betweenness().as_clustering()
#         num_com = len(com)
#         # palette = ig.ClusterColoringPalette(n=num_com)
#         palette = ig.AdvancedGradientPalette(["pink", 'palegoldenrod',"palegreen", "powderblue"], n=num_com)
#         for i, community in enumerate(com):
#             ssg.vs[community]["color"] = i
#             community_edges = ssg.es.select(_within=community)
#             community_edges["color"] = i
        
# # transfer edge weights from bidirectional graph

#         for i,e in enumerate(ssg.es):
#             ss = e.source
#             tt = e.target
#             for f in subg.es:
#                 if f.source == ss and f.target == tt:
#                     ssg.es[i]['combined_score'] = f['combined_score']
            
            
# # get edge values

#         pvals = [p for p in ssg.es['combined_score']]
#         pmax = max(pvals)
#         pmin = min(pvals)
#         prange = pmax-pmin
        
#         fig,ax = plt.subplots(figsize=(10,10))
#         ax.set_title(f'Drug : {d} - Target : {targ}', fontsize=16)
        
#         # ssg.vs['label'] = ssg.vs['name']
#         ssg.vs['label'] = [x if x not in top10 else f'{top10.index(x)}\n{x}' for x in ssg.vs['name']]
#         lay = ssg.layout('kk')
#         style = {}
#         style['layout'] = lay
#         # style['autocurve'] = True
#         # style['bbox'] = (1024,1024)
#         # style['margin'] = 50
#         style['vertex_size'] = [500 if n == targ else 350 if n in top10 else 250 for n in ssg.vs['name']]
#         style['vertex_color'] = ['pink' if n == targ else 'moccasin' if n in top10 else 'white' for n in ssg.vs['name']]
#         # style['edge_arrow_size'] = 0.75
#         # #style['edge_width'] = [min([p,10]) for p in gg.es['p']]
#         style['edge_width'] = [10*(p-pmin)/prange+2.0 for p in pvals]
#         # style['edge_arrow_size'] = [max(0.5,2.2*(p-pmin)/prange+0.1) for p in ssg.es['combined_score']]

#         ig.plot(com,palette=palette,target=ax,**style)
#         # print(targ+'_net2.pdf created' )
#         # plt.show()

    
    
    
    
    
    
    
    
    
    
        
        

