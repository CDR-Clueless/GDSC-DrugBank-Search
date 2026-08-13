#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 11 Jun 2026

@author: jds40
"""
import os
from copy import deepcopy
from typing import Optional, Union
from collections.abc import Callable
import multiprocessing as mp

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy.stats import linregress
import json

from drug_search import get_data
from logger import Logger

CLEANED_DATA_DIR: str = os.path.join("Data", "Laurence-Data")
DEFAULT_CRISPR_FILE: str = os.path.join(CLEANED_DATA_DIR,"CRISPRGeneDependency.csv")
DEFAULT_HUGO_FILE: str = os.path.join(CLEANED_DATA_DIR, "hgnc_complete_set.tsv")
DEFAULT_CELL_INFO_FILE: str = os.path.join(CLEANED_DATA_DIR, "Model.csv")
DEFAULT_DRUG1_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC1_drug_results_target_cleaned7.tsv')
DEFAULT_DRUG2_FILE: str = os.path.join(CLEANED_DATA_DIR, 'GDSC2_drug_results_target_cleaned7.tsv')
DEFAULT_DRUG_COMB_FILE: str = os.path.join("Data", "Raw Data", "GDSCC")

# Decide whether this should be a debug mode
if(not os.path.exists(os.path.join("Local", "localVars.json"))):
    DEBUG_MODE: bool = False
else:
    with open(os.path.join("Local", "localVars.json"), "rb") as f:
        DEBUG_MODE: bool = json.load(f)["DEBUG_MODE"]

def dummyLinear(x1: float = 1.0, c: float = 0.1, noise: float = 0.1, lower: float = 0.0, upper: float = 1.0, n: int = 100) -> np.ndarray:
    return (np.linspace(lower, upper, n) * x1) + c + (noise * np.random.uniform(0, 1, n))

def dummyTwo(x1: float = 1.0, x2: float = 1.0, c: float = 0.1, noise: float = 0.1, lower: float = 0.0, upper: float = 1.0, n: int = 100) -> np.ndarray:
    base = np.linspace(lower, upper, n)
    return (base * x1) + (x2 * np.power(base, 2.)) + c + (noise * np.random.uniform(lower, upper, n))

def dummyX(xs: list = [1, 2], c: float = 0.1, noise: Union[list, np.ndarray, float] = 0.1, lower: float = 0.0, upper: float = 1.0, n: int = 100) -> tuple[np.ndarray]:
    base = np.linspace(lower, upper, n)
    # If noise is an array, make it the same shape 
    if(noise is not float and noise is not int):
        realNoise = np.ones(n)
        for i in range(len(noise)):
            fracl, fracu = i/len(noise), (i+1)/len(noise)
            realNoise[int(fracl*realNoise.shape[0]):int(fracu*realNoise.shape[0])] = noise[i]
        # Fill in the remaining noise
        realNoise[int(fracu*realNoise.shape[0]):] = noise[-1]
        noise = realNoise
    y = (np.random.uniform(-1, 1, n) * noise) + c
    for power, x in enumerate(xs):
        y += x * np.power(base, power+1)
    return np.vstack([np.ones(n, dtype = float)] + [np.power(base, power+1) for power in range(len(xs))]).T, y
    

def main():
    testing()
    return
    #calculate_survCorr()
    #return
    # Generate some dummy data
    x, y = dummyX()

    # Try calculating SC using this
    x = x[:,1]
    rsq, eq = calculate_SC_GLS(x, y, True, 2)

    plt.scatter(x, y)
    plt.plot(x, [eq(xi) for xi in x], color = "black")
    plt.text(0.5, 0.5, calculate_rsquared(x, y, 2))
    plt.show()

    return

def testing():
    true = np.array([0.1, 4], dtype = float)
    x, y = dummyX(xs = true[1:], c = true[0], upper = 100, noise = [750, 250, 500, 5, 700])
    # Fit models
    gls_model = sm.GLS(y, x)
    resGLS = gls_model.fit()
    rhoGLS = resGLS.params
    wlsD_rsq, rhowlsD = calculate_SC_WLS_data(x[:,1], y, components=1, return_coefficients=True)
    wlsP_rsq, rhowlsP = calculate_SC_WLS_prediction(x[:,1], y, components=1, return_coefficients=True)
    
    plt.scatter(x[:, 1], y)
    plt.plot(x[:, 1], sum([x[:,i]*true[i] for i in range(len(true))]), label = "True", color = "blue")
    plt.plot(x[:, 1], sum([x[:,i]*rhoGLS[i] for i in range(len(rhoGLS))]), label = "GLS", color = "orange")
    plt.plot(x[:, 1], sum([x[:,i]*rhowlsD[i] for i in range(len(rhowlsD))]), label = "WLS Data", color = "green")
    plt.plot(x[:, 1], sum([x[:,i]*rhowlsP[i] for i in range(len(rhowlsP))]), label = "WLS GLS-Error", color = "red")
    plt.legend()
    print(f"GLS Error: {true - rhoGLS}\nWLS Data Error: {true - rhowlsD}\nWLS GLS Error: {true - rhowlsP}")
    plt.show()
    return

def calculate_rsquared(x, y, nComponents: int = 2, return_equation: bool = False, return_coefficients: bool = False) -> float:
    return calculate_SC_GLS(x, y, components = nComponents, return_equation = return_equation, return_coefficients = return_coefficients)

def calculate_survCorr(crisprDepsLoc: Optional[str] = None, hugoLoc: Optional[str] = None, cellInfoLoc: Optional[str] = None,
         gdsc1Loc: Optional[str] = None, gdsc2Loc: Optional[str] = None,
         logFile: Logger = Logger(os.path.join("Data", "Results", "GDSC-SC-calculation-output.txt")),
         correlation_mode: str = "pearson", GLM_components: int = 1, dMode: bool = DEBUG_MODE):
    # First, update logFile path if in debug mode
    if(dMode):
        logFile.directory = logFile.directory[:-4]+"-DEBUG"+logFile.directory[-4:]
    # Clear LogFile and add initial line
    logFile.clear()
    logFile.add("GDSC Function starts")

    # Set up function to calculate the Survivability Correlation
    correlation_mode = correlation_mode.lower().replace(" ","")
    correlation_calculator: Callable = {"pearson": calculate_SC_Pearson}[correlation_mode]
    # Compile dictionary of relevant file locations
    fileLocs = {}
    for i, name in enumerate(["crispr", "hugo", "cellinfo", "gdsc1", "gdsc2"]):
        alternative = [crisprDepsLoc, hugoLoc, cellInfoLoc, gdsc1Loc, gdsc2Loc][i]
        if(alternative is not None):
            fileLocs[name] = alternative
        else:
            fileLocs[name] = [DEFAULT_CRISPR_FILE, DEFAULT_HUGO_FILE, DEFAULT_CELL_INFO_FILE, DEFAULT_DRUG1_FILE, DEFAULT_DRUG2_FILE][i]

    # Get number of CPU's to use for multiprocessing
    cpu_count = max(1, mp.cpu_count()-2)

    # Get known CRISPR cell line-gene dependencies (row index = model ID/cell line ID, column = Gene)
    crisprDeps = pd.read_csv(fileLocs["crispr"]).fillna(0.0)

    crisprDeps.rename(columns = {'Unnamed: 0':'ModelID'},inplace=True)
    crisprDeps.set_index('ModelID', inplace=True)
    
    # edit header names to remove spaces etc.
    gg = dict(zip(list(crisprDeps.columns), [i.strip().split()[0]
              for i in list(crisprDeps.columns)]))
    crisprDeps.rename(columns=gg, inplace=True)

    # Get HUGO standardised gene name information
    hgnc = pd.read_table(fileLocs["hugo"], low_memory=False).fillna('')
    hgnc = hgnc[['symbol', 'ensembl_gene_id',
                 'prev_symbol', 'location', 'location_sortable']]
    hgnc.set_index('symbol', inplace=True)
    
    # Correct legacy gene names in crisprDeps using HUGO table
    old_names = set(crisprDeps.columns) & (set(hgnc.index) ^ set(crisprDeps.columns))
    logFile.add(f'Updating {len(old_names)} archaic gene names in dependency data')
    for g_old in old_names:
        g_new = hgnc[hgnc['prev_symbol'].str.contains(g_old)].reset_index()['symbol']
        if len(g_new) == 0 or (g_new[0] not in hgnc.index):
            continue
        else:
            crisprDeps.rename(columns={g_old: g_new[0]}, inplace=True)
    
    # Load in cell line information - useful for linking cell line names/IDs/etc. from other DataFrames to other information from other DataFrames
    clInfo = pd.read_csv(fileLocs["cellinfo"], low_memory=False).fillna('')
    clInfo['OncotreeLineage'] = [x.upper() for x in clInfo['OncotreeLineage']]
    clInfo["OncotreePrimaryDisease"] = clInfo["OncotreePrimaryDisease"].str.replace(' ','_')

    # Separate out cancer types from cell line information as it is of particular note
    cancer_types = set(clInfo['OncotreeLineage'])

    # Load in GDSC data
    drug1 = pd.read_table(fileLocs["gdsc1"], low_memory=False).fillna('')
    drug1["DRUG_NAME"] = drug1["DRUG_NAME"].apply(lambda x:x.upper())
    
    drug2 = pd.read_table(fileLocs["gdsc2"], low_memory=False).fillna('')
    drug2["DRUG_NAME"] = drug2["DRUG_NAME"].apply(lambda x:x.upper())

    # Initilise drug by gene data
    dList = sorted(set(drug1["DRUG_NAME"]) | set(drug2["DRUG_NAME"]))

    for d in dList:
        # Get all available cell lines
        cs = [df[df["DRUG_NAME"]==d] for df in [drug2, drug1]]
        
        # Go through all genes from CRISPR dependencies DataFrame
        genes = crisprDeps.columns
        # If in Debug mode, only do this for the first 20 genes
        if(dMode):
            genes = genes[:20]
        for i, gn in enumerate(genes):
            
            # get dependencies (deps) for all available cell lines, as well as a list of cell lines which
            # were found within the deps DataFrame

            deps = [crisprDeps[crisprDeps.index.isin(cldf["ModelID"].values)][gn].reset_index() for cldf in cs]
            #print(deps)
            dep_names = []
            for cldf in deps:
                if(len(cldf)>0):
                    dep_names.append(list(cldf["ModelID"]))
                else:
                    dep_names.append(None)
            #dep_names = [list(cldf["ModelID"]) if len(cldf) > 0 else None for cldf in cs ]

            # Get DataFrames of whatever is being used for establishing correlations (i.e. pKi, IC50 or eMax values)
            responses = []
            for gdscdf, cldf, cldf_names in zip(cs, deps, dep_names):
                if(len(cldf)<1):
                    responses.append(None)
                    continue
                newResponse = gdscdf[(gdscdf["DRUG_NAME"] == d) & 
                                (gdscdf["ModelID"].isin(cldf_names))].drop_duplicates \
                            (subset=["ModelID"], keep="first")[["LN_IC50","ModelID"]]
                responses.append(cldf.merge(newResponse))

            # Get Survivability Correlation and plot it
            for gdscv, response in zip(["GDSC2", "GDSC1"], responses):
                if(response is None):
                    continue
                
                # Get the two different responses - which is the Cell Line dependencies and the "LN_IC50" (eMax, IC50 etc.) data
                x = np.array(response[gn])
                y = response["LN_IC50"]
                # If dealing with IC50 data, multiply is by -1 to transform it from LN(IC50) to pIC50 (pIC50 = -LN(IC50))
                if("IC50" in "LN_IC50".upper()):
                    y *= -1
                # If dealing with eMax data, take the natural log and multiply by -1 to obtain peMax (-Ln(eMax)) from eMax values
                elif("emax" in "LN_IC50".lower()):
                    y = -1 * np.log(y)
                
                # Find the predictability and equation
                pr, eq = correlation_calculator(x, y, True)

                plt.scatter(x, y)
                plt.plot([min(x), max(x)], [eq(min(x)), eq(max(x))], linestyle = "--", color = "black")
                plt.text(max(x)-((max(x)-min(x))/10), eq(max(x))+((max(y)-min(y))*.05), f"{pr:.2f}")
                plt.title(f"{d}-{gn} responses")
                plt.xlabel("Cell Line CRISPR Dependency")
                plt.ylabel("pIC50")
                plt.show()
                if(DEBUG_MODE):
                    return

def calculate_SC_Pearson(x, y, return_equation: bool = False, components: int = 1) -> float | tuple[float, Callable]:
    result = linregress(x, y)
    m, c, pr, pp, mErr, cErr = result.slope, result.intercept, result.rvalue, result.pvalue, result.stderr, result.intercept_stderr
    if(not return_equation):
        return pr
    else:
        return (pr, lambda x: (m*x) + c)

def calculate_SC_GLS(x, y, components: int = 1, return_equation: bool = False, return_coefficients: bool = False) -> Union[float, tuple[float, Callable], tuple[float, float], tuple[float, Callable, float]]:
    """Calculate Generalized Least Squares regression between data x and y

    Args:
        x (_type_): Input data
        y (_type_): Output data
        return_equation (bool, optional): Whether to return a function which serves as the equation of the fitted GLS model. Defaults to False.
        components (int, optional): Maximum power of x (i.e. fitted esimator equation = c + m1x + m2x^2 + ... + m{components}x^{components}). Defaults to 1.

    Returns:
        float | tuple[float, Callable]: Returns r^2 value of the fitted model and, optionally, a function serving as an equation to the fitted line of the model
    """
    # First, format x into a form interpretable by a GLM with the desired component count
    formattedX = np.vstack([np.ones(x.shape[0], dtype = float)] + [np.power(x, power+1) for power in range(components)]).T
    # Now fit the model and extract relevant parameters
    lin_model = sm.GLS(y, formattedX)
    res = lin_model.fit()
    rho = res.params
    pr = res.rsquared
    if(not return_equation and not return_coefficients):
        return pr
    output = [pr]
    if(return_equation):
        output.append(lambda result: rho[0] + np.dot(create_squared_array(result, rho.shape[0]-1), rho[1:].T))
    if(return_coefficients):
        output.append(rho)
    return deepcopy(tuple(output))

def calculate_SC_WLS(x, y, components: int = 1, return_equation: bool = False, return_coefficients: bool = False,
                          noiseMethod: str = "data", noiseSections: int = 10, noiseSplit: str = "values") -> Union[float, tuple[float, Callable], tuple[float, float], tuple[float, Callable, float]]:
    """Calculate Weighted Least Squares regression between data x and y

    Args:
        x (_type_): Input data
        y (_type_): Output data
        return_equation (bool, optional): Whether to return a function which serves as the equation of the fitted GLS model. Defaults to False.
        components (int, optional): Maximum power of x (i.e. fitted esimator equation = c + m1x + m2x^2 + ... + m{components}x^{components}). Defaults to 1.
        noiseSections (int, optional): Number of sections of data with different noise values. Defaults to 10
        noiseSplit (str, optional): Determines whether noise sections should be determined by data points (i.e. sections of equal size) or values (i.e. noise x1 for 0.0-0.1, x2 for 0.1-0.2, etc.)

    Returns:
        float | tuple[float, Callable]: Returns r^2 value of the fitted model and, optionally: a function serving as an equation to the fitted line of the model; resultant coefficients of the model
    """
    # First, sort the x and y values by their x value for easier manipulation
    allSort = sorted(zip(x, y))
    x, y = np.array([e[0] for e in allSort], dtype = float), np.array([e[1] for e in allSort], dtype = float)
    # Create a formatted version of X interpretable by statsmodels now to save duplicate work later
    formattedX = np.vstack([np.ones(x.shape[0], dtype = float)] + [np.power(x, power+1) for power in range(components)]).T
    ## Next, get an array representing the noise weights to feed into the WLS model
    # use the raw data points if noiseMethod is set to 'date', or the errors resulting from a trained GLS if noiseMethod is set to 'error'
    if(noiseMethod.lower().strip()=="data"):
        noise = create_wls_weights(x, y, noiseSections, noiseSplit)
    else:
        ## Start by training an appropriate GLS model
        # Fit the model and extract relevant parameters
        gls_model = sm.GLS(y, formattedX)
        res = gls_model.fit()
        rho = res.params
        ## Now calculate all the errors of this model
        # Get predicted y values of model
        predictions = np.dot(rho, formattedX.T)
        # Subtract predictions from true values to obtain error
        errors = y - predictions
        # Finally, create the weights required using these errors
        noise = create_wls_weights(x, errors, noiseSections, noiseSplit)
            
    # Now fit the model and extract relevant parameters
    wls_model = sm.WLS(y, formattedX, weights = 1 / (noise**2))
    res = wls_model.fit()
    rho = res.params
    pr = res.rsquared
    if(not return_equation and not return_coefficients):
        return pr
    output = [pr]
    if(return_equation):
        output.append(lambda result: rho[0] + np.dot(create_squared_array(result, rho.shape[0]-1), rho[1:].T))
    if(return_coefficients):
        output.append(rho)
    return deepcopy(tuple(output))

def calculate_SC_WLS_data(x, y, components: int = 1, return_equation: bool = False, return_coefficients: bool = False,
                          noiseSections: int = 10, noiseSplit: str = "values") -> Union[float, tuple[float, Callable], tuple[float, float], tuple[float, Callable, float]]:
    """Calculate Weighted Least Squares regression between data x and y using raw data to determine weights

    Args:
        x (_type_): Input data
        y (_type_): Output data
        return_equation (bool, optional): Whether to return a function which serves as the equation of the fitted GLS model. Defaults to False.
        components (int, optional): Maximum power of x (i.e. fitted esimator equation = c + m1x + m2x^2 + ... + m{components}x^{components}). Defaults to 1.
        noiseSections (int, optional): Number of sections of data with different noise values. Defaults to 10
        noiseSplit (str, optional): Determines whether noise sections should be determined by data points (i.e. sections of equal size) or values (i.e. noise x1 for 0.0-0.1, x2 for 0.1-0.2, etc.)

    Returns:
        float | tuple[float, Callable]: Returns r^2 value of the fitted model and, optionally, a function serving as an equation to the fitted line of the model
    """

    return calculate_SC_WLS(x, y, components, return_equation, return_coefficients, "data", noiseSections, noiseSplit)

def calculate_SC_WLS_prediction(x, y, components: int = 1, return_equation: bool = False, return_coefficients: bool = False,
                          noiseSections: int = 10, noiseSplit: str = "values") -> Union[float, tuple[float, Callable], tuple[float, float], tuple[float, Callable, float]]:
    """Calculate Weighted Least Squares regression between data x and y using GLS prediction error to determine weights

    Args:
        x (_type_): Input data
        y (_type_): Output data
        return_equation (bool, optional): Whether to return a function which serves as the equation of the fitted GLS model. Defaults to False.
        components (int, optional): Maximum power of x (i.e. fitted esimator equation = c + m1x + m2x^2 + ... + m{components}x^{components}). Defaults to 1.
        noiseSections (int, optional): Number of sections of data with different noise values. Defaults to 10
        noiseSplit (str, optional): Determines whether noise sections should be determined by data points (i.e. sections of equal size) or values (i.e. noise x1 for 0.0-0.1, x2 for 0.1-0.2, etc.)

    Returns:
        float | tuple[float, Callable]: Returns r^2 value of the fitted model and, optionally, a function serving as an equation to the fitted line of the model
    """

    return calculate_SC_WLS(x, y, components, return_equation, return_coefficients, "prediction", noiseSections, noiseSplit)

def create_wls_weights(x, y, noiseSections: int = 10, noiseSplit: str = "values") -> np.ndarray[tuple[()],float]:
    """ Create an array which can be plugged into WLS models as weights = 1 / (output**2)

    Args:
        x (_type_): _description_
        y (_type_): _description_
        noiseSections (int, optional): Number of sections of data with different noise values. Defaults to 10.
        noiseSplit (str, optional): Determines whether noise sections should be determined by counts of data points (i.e. sections of equal size) or values (i.e. noise x1 for 0.0-0.1, x2 for 0.1-0.2, etc.). Defaults to "values".

    Returns:
        np.ndarray: Array of values which can be plugged into WLS weights as weights = 1 / (np.ndarray**2)
    """
    # First, sort the x and y values by their x value for easier manipulation
    allSort = sorted(zip(x, y))
    x, y = np.array([e[0] for e in allSort], dtype = float), np.array([e[1] for e in allSort], dtype = float)
    ## Next, get an array representing the noise weights to feed into the WLS model
    noise = np.zeros(x.shape[0], dtype = float)
    # Get noise values from equally sized sections
    if(noiseSplit.lower().strip() in ["counts", "count"]):
        # Fraction of the total each section needs to be
        frac = 1/noiseSections
        # Assign noise values based on standard deviation of y values
        for i in range(1, noiseSections+1):
            # Set upper and lower bounds
            lb, ub = int((i-1)*frac*x.shape[0]), int(i*frac*x.shape[0])
            val = np.std(y[lb:ub])
            noise[lb:ub] = val
        # Ensure the final noise entry has a value
        if(noise[-1]==0.0):
            noise[-1] = val
    # Get noise values from sections between certain values
    else:
        # Get upper and lower bounds for values from which sections are derived
        # Note that x[0] and x[-1] can be used instead of minimum and maximum as they've already been sorted
        lb, ub = x[0], x[-1]
        # Create an array from lb to ub, these will determine cutoff points to use for groups from which standard deviations are derived
        cutoffs = np.linspace(lb, ub, noiseSections+1, endpoint=True)
        cMark, j = 1, 0 # Counters to mark current cutoff used in cutoffs, maximum index of previous cutoff + 1
        for i in range(x.shape[0]):
            # Check if the current sample has gone above the relevant cutoff point;
            # calculate the relevant standard deviation, use it as noise, and reset j index if so
            if(x[i]>cutoffs[cMark]):
                noise[j:i] = np.std(y[j:i])
                j = i
                # While is used here to skip any empty sections (i.e. if 0.5-0.6 is a section but there are no values there)
                while(x[i]>cutoffs[cMark]):
                    cMark += 1
        # Ensure the final noise values are filled
        if(j<noise.shape[0]):
            noise[j:] = np.std(y[j:])

    return noise

def create_squared_array(x, shape: int = 2, dtype = float) -> np.ndarray:
    """ Create an array of [x, x^2, x^3, ..., x^{shape}]

    Args:
        x (_type_): Base number
        shape (int, optional): Length of array and maximum power to calculate. Defaults to 2.
        dtype (_type_, optional): Data type to make the array; accepts integer or floating point number data types. Defaults to float.

    Returns:
        np.ndarray: _description_
    """
    output = np.full(shape, x, dtype = dtype)
    for i in range(1, shape):
        output[i] = np.power(output[0] ,i+1)
    return output

if(__name__=="__main__"):
    main()