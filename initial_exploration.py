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
from scipy.optimize import curve_fit
import seaborn as sns
from copy import deepcopy
import json
from tqdm import tqdm
import ast
import igraph as ig
import diptest

from typing import Optional

from data_handler import DataHandler
from searcher import Searcher
from drug_gene_correlation_histograms import CorrelationPlotter, curve_guess
from drug_search import update_hgnc, get_data

CLEANED_DATA_DIR: str = os.path.join("Data", "Laurence-Data")

DEFAULT_HUGO_FILE: str = os.path.join(CLEANED_DATA_DIR, "hgnc_complete_set.tsv")
DEFAULT_ALL_BY_ALL_FILE: str = os.path.join(CLEANED_DATA_DIR, "AllDrugsByAllGenes.tsv")
DEFAULT_STRING_INFO_FILE: str = os.path.join(CLEANED_DATA_DIR, "9606.protein.info.v12.0.txt")
DEFAULT_STRING_LINK_FILE: str = os.path.join(CLEANED_DATA_DIR, "9606.protein.links.v12.0.ssv")

TARGET_DRUG: str = "965-D2"
START_POINT_CUTOFF: float = 0.197

def gaussian(x, A, mu, sigma):
    return A*np.exp(-np.divide(np.power(x-mu, 2),(2*np.power(sigma, 2))))

def bimodal(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2)

def main():

    data = CorrelationPlotter().datasets
    data = data["AllByAll"]
    data = data.set_index("Drug")
    ### Fitting a gaussian
    for i in range(len(data)):
        dist = data.iloc[i].values
        dt = diptest.diptest(dist)
        if(dt[1]>0.4):
            print(f"Result for number {i}, drug {data.iloc[i].name}: {dt[1]}")
            ind, drug, dist = i, data.iloc[i].name, dist
            break
    
    counts, bins = np.histogram(dist, bins = np.arange(-1, 1.025, 0.025))
    xs = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    plt.stairs(counts, bins)
    ## Initial guess for a and mu values of gaussian - A is the amplitude of a gaussian and mu is the point at which that amplitude is reached
    iga = np.max(counts)
    igmu = xs[np.where(counts==iga)[0][0]]
    ## Fit curve
    #params, cov = curve_fit(gaussian, xs, counts, p0 = np.array([iga, igmu, 0.1])) #(np.mean(dist), max(counts), 0.5))
    params, cov = curve_fit(gaussian, xs, counts, p0 = curve_guess(xs, counts, "gaussian"))
    #print(params)
    fitted_xs = np.linspace(xs[0], xs[-1], len(xs)*100)
    plt.plot(fitted_xs, gaussian(fitted_xs, *params), color = "r")
    plt.title("Gaussian-fitted distribution")
    plt.show()

    ### Fitting a bimodal
    for i in range(len(data)):
        dist = data.iloc[i].values
        dt = diptest.diptest(dist)
        if(dt[1]<0.05):
            print(f"Result for number {i}, drug {data.iloc[i].name}: {dt[1]}")
            ind, drug, dist = i, data.iloc[i].name, dist
            break
    
    counts, bins = np.histogram(dist, bins = np.arange(-1, 1.025, 0.025))
    xs = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    plt.stairs(counts, bins)
    ## Initial guess for the a and mu values of the two bimodals (a1 is the first mode amplitude, mu1 the first mode position, a2,mu2 the second)
    peaks = [(xs[i], counts[i]) for i in range(1, len(xs)-1) if (counts[i-1]<counts[i] and counts[i+1]<counts[i])]
    # Remove extra peaks
    if(len(peaks)>2):
        t, st = (0, -np.inf), (0, -np.inf)
        for tup in peaks:
            if(tup[1]>t[1]):
                t = deepcopy(tup)
            elif(tup[1]>st[1]):
                st = deepcopy(tup)
        peaks = [t, st]
    ## Curve fitting
    #params, cov = curve_fit(bimodal, xs, counts, p0 = np.array([peaks[0][1], peaks[0][0], 0.05, peaks[1][1], peaks[1][0], 0.05], dtype = float)) #(np.mean(dist), max(counts), 0.5))
    print(f"Bimodal guess = {curve_guess(xs, counts, "bimodal")}")
    params, cov = curve_fit(bimodal, xs, counts, p0 = curve_guess(xs, counts, "bimodal"))
    #print(params)
    fitted_xs = np.linspace(xs[0], xs[-1], len(xs)*100)
    plt.plot(fitted_xs, bimodal(fitted_xs, *params), color = "r", label = "fitted")
    #plt.plot(fitted_xs, bimodal(fitted_xs, peaks[0][0], 0.02, peaks[0][1], peaks[1][0], 0.02, peaks[1][1]), label = "guess")
    plt.legend()
    plt.title("Bimodal-fitted distribution")
    plt.show()
    return
    

    """
    ## Import relevant datasets and amend them
    # HGNC
    hgnc = pd.read_table(DEFAULT_HUGO_FILE, low_memory=False).fillna('')
    hgnc = hgnc[['symbol', 'ensembl_gene_id',
                'prev_symbol', 'location', 'location_sortable']]
    hgnc.set_index('symbol', inplace=True)

    # STRING proteins
    stinfo =  pd.read_table(DEFAULT_STRING_INFO_FILE,
                            usecols=['#string_protein_id','preferred_name']).fillna('')
    stinfo.rename(columns = {'#string_protein_id':'ID','preferred_name':'symbol'},inplace=True)
    stinfo = update_hgnc(stinfo, hgnc)

    stdict = dict(zip(stinfo.ID,stinfo.symbol))

    # STRING network
    stlink = pd.read_csv(DEFAULT_STRING_LINK_FILE,sep=' ')
    stlink.combined_score = stlink.combined_score.astype(int)
    # Change proteins in stlink from ID's to HGNC-checked names
    stlink.protein1 = stlink.protein1.apply(lambda x: stdict[x])
    stlink.protein2 = stlink.protein2.apply(lambda x: stdict[x])

    # DrugxGene survivability scores
    allbyall = pd.read_csv(DEFAULT_ALL_BY_ALL_FILE, sep = "\t")
    allbyall = update_hgnc(allbyall, hgnc)
    allbyall = allbyall.set_index("symbol")

    # Refine STRING links to those with a combined score >0.8
    stlink[stlink.combined_score.gt(800)]
    
    print('Building human link graph ...')
    g = ig.Graph.TupleList(stlink.itertuples(index=False), directed=True, weights=False, edge_attrs="combined_score")
    print('Done !!')
    g.vs["survivability"] = [allbyall[TARGET_DRUG].loc[gn] if gn in allbyall[TARGET_DRUG].index else float("NaN") for gn in g.vs["name"]]
    print(ig.summary(g))
    startPoints = [n for s,n in zip(g.vs["survivability"], g.vs["name"]) if s > START_POINT_CUTOFF]
    print(startPoints)
    shortest = np.inf
    for sp in startPoints:
        shortpath = g.get_shortest_paths(sp, to="MKNK1", weights=g.es["combined_score"], output="vpath")[0]
        shortest = min(shortest, len(shortpath)-1)
    
    #USE HGNC ON DRUGBANK COMPARISON OUTPUT AND EXTEND SHORTEST PATHFINDING TO ALL TARGETS
    """
    return

if __name__ == "__main__":
    main()
