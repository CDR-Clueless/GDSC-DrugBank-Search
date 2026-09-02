#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 27 Aug 2026

@author: jds40
"""

from typing import Union, Optional
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import stats

def cohenD(data: Union[tuple, list, pd.DataFrame, np.ndarray]) -> float:
    # Ensure data is properly fomratted
    data = format_2col(data, fill = True)
    if(type(data)==str):
        return np.nan
    # Get the means and pooled standard deviation
    m1, m2 = np.nanmean(data[:,0]), np.nanmean(data[:,1])
    pSD = pooledSD(data)
    return abs(np.divide(m1 - m2, pSD))

def format_2col(data: Union[tuple, list, pd.DataFrame, np.ndarray], fill: bool = False) -> Union[np.ndarray, str]:
    # Ensure data is a numpy array
    if(type(data) == np.ndarray):
        formatted = data
    else:
        if(type(data) == pd.DataFrame):
            formatted = data.to_numpy()
        elif(type(data) in [tuple, list]):
            # Fill if desired
            if(fill):
                toformat, shapeN = [], max([len(l) for l in data])
                for row in data:
                    toformat.append(deepcopy(list(row) + [np.nan for i in range(shapeN - len(row))]))
            else:
                toformat = data
            formatted = np.array(toformat, dtype = float)
    # Ensure data is an (n, 2) matrix
    if(formatted.shape[0] == 2 and formatted.shape[1] != 2):
        formatted = formatted.T
    elif(formatted.shape[0] != 2 and formatted.shape[1] != 2):
        return f"Unsupported dimensions: {formatted.shape}. This function specifically requires 2 columns or rows"
    return formatted

def pooledSD(data: Union[tuple, list, pd.DataFrame, np.ndarray]) -> float:
    # Ensure data is formatted
    data = format_2col(data)
    # Extract the two groups
    g1, g2 = data[:,0], data[:,1]
    g1, g2 = g1[~np.isnan(g1)], g2[~np.isnan(g2)]
    # Calculate standard deviation and sample size for each group
    s1, s2 = np.std(g1), np.std(g2)
    n1, n2 = g1.shape[0], g2.shape[0]
    # Prepare numerator and denominator (this could be done in 1 line, but would be less clear)
    num = ((n1-1)*np.power(s1, 2.)) + ((n2-1)*np.power(s2, 2.))
    den = n1 + n2 - 2
    return np.sqrt(np.divide(num, den))

def chisquare_hom(data: Union[tuple, list, pd.DataFrame, np.ndarray], minimum: int = 0) -> float:
    # Ensure data is properly formatted
    # Ensure data is properly fomratted
    data = format_2col(data)
    if(type(data)==str):
        return np.nan
    # Ensure all rows have the minimum value in at least 1 sample
    if(minimum > 0):
        data = data[np.logical_or(data[:,0] >= minimum, data[:,1] >= minimum)]
    # Calculate array of expected variables ((Row Total * Col Total) / Grand total)
    rowTot = np.sum(data, axis = 1, dtype = float)
    colTot = np.sum(data, axis = 0, dtype = float)
    e = np.dot(np.reshape(rowTot, (rowTot.shape[0], 1)), np.reshape(colTot, (1, colTot.shape[0]))) / np.sum(data)
    chimat = np.divide(np.power(data - e, 2.), e)
    chi = np.sum(chimat)
    df = (data.shape[0]-1) * (data.shape[1]-1)
    p = 1. - stats.chi2.cdf(chi, df)
    return p

def main():
    test = np.random.randint(low = 0, high = 10, size = (20, 2))
    print(test)
    chisquare_hom(test, minimum = 5)
    return

if(__name__=="__main__"):
    main()
