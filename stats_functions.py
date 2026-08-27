#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 27 Aug 2026

@author: jds40
"""

from typing import Union, Optional

import numpy as np
import pandas as pd

def cohenD(data: Union[tuple, list, pd.DataFrame, np.ndarray]) -> Optional[float]:
    # Ensure data is properly fomratted
    data = format_2col(data)
    # Get the means and pooled standard deviation
    m1, m2 = np.mean(data[:,0]), np.mean(data[:,1])
    pSD = pooledSD(data)
    return abs(np.divide(m1 - m2, pSD))

def format_2col(data: Union[tuple, list, pd.DataFrame, np.ndarray]) -> Union[np.ndarray, str]:
    # Ensure data is a numpy array
    if(data is np.ndarray):
        formatted = data
    else:
        if(data is pd.DataFrame):
            formatted = data.to_numpy()
        elif(data is tuple or data is list):
            formatted = np.array([np.array(data[i], dtype = float) for i in range(len(data))], dtype = float)
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
    g1, g2 = data[:,0].dropna(), data[:,1].dropna()
    # Calculate standard deviation and sample size for each group
    s1, s2 = np.std(g1), np.std(g2)
    n1, n2 = g1.shape[0], g2.shape[0]
    # Prepare numerator and denominator (this could be done in 1 line, but would be less clear)
    num = ((n1-1)*np.power(s1, 2.)) + ((n2-1)*np.power(s2, 2.))
    den = n1 + n2 - 2
    return np.sqrt(np.divide(num, den))

def main():
    return

if(__name__=="__main__"):
    main()
