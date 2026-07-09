#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2 Jul 2026

@author: jds40
Adapts "combine_subscores.py" code from STRING database
"""

from __future__ import print_function
import os
import sys
import argparse
import multiprocessing as mp

import numpy as np

from logger import Logger

INPUT_FILE: str = os.path.join("Data", "Raw Data", "STRING", "9606.protein.links.full.v12.0.txt")
OUTPUT_FILE: str = os.path.join("Data", "Calculated Data", "Custom.String.Calculation.tsv")

##########################################################
## This script combines all the STRING's channels subscores
## into the final combined STRING score.
## It uses unpacked protein.links.full.xx.txt.gz as input
## which can be downloaded from the download subpage:
##      https://string-db.org/cgi/download.pl
##
## Algorithm updated for versions v12 and up.
##########################################################

def main(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    banned_methods = args.banned_methods.split(",")
    desired_methods: list = args.desired_methods.split(",")
    logger = Logger(args.logfileLoc)
    calculate(banned_methods, args.sourceFile, args.destinationFile, desired_methods, max(mp.cpu_count()-2, 1), logger)
    return

def calculate(banned_methods: list = ["textmining"], inputFile: str = INPUT_FILE, outputFile: str = OUTPUT_FILE,
              desired_methods: list =[], coreCount: int = max(mp.cpu_count()-2, 1), logFile: Logger = Logger()):
    
    logFile.clear()
    logFile.add(f"Banned methods: {banned_methods}\nDesired methods: {desired_methods}\nInput File: {inputFile}\nOutput File: {outputFile}")

    # Format banned_methods to standardise for code comparison
    banned_methods = [s.lower().replace(" ","") for s in banned_methods]
    desired_methods = [s.lower().replace(" ","") for s in desired_methods]

    if not os.path.exists(inputFile):
        logFile.add("Can't locate input file %s" % INPUT_FILE)
        print("Can't locate input file %s" % INPUT_FILE)
        return

    prior = 0.041

    # Iterate over lines of the given file; skip the first as it's the header line
    with open(inputFile, "r") as f:
        header = True
        for line in f:
            if(header):
                header = False
                continue
            calculate_gene(line, prior, banned_methods, desired_methods, outputFile)
    
    logFile.add("Finished STRING calculation")

    #mp.Pool(coreCount).starmap_async(calculate_gene,
                #[(line, prior, banned_methods, desired_methods, outputFile)
                #for line in range(lines)]).get()

def compute_prior_away(score, prior):

    if score < prior: score = prior
    score_no_prior = (score - prior) / (1 - prior)

    return score_no_prior

def calculate_gene(line, prior: float, banned_methods: list, desired_methods: list, outputFile: str):
    l = line.split()
    
    ## load the line
        
    (protein1, protein2,
    neighborhood, neighborhood_transferred,
    fusion, cooccurrence,
    homology,
    coexpression, coexpression_transferred,
    experiments, experiments_transferred,
    database, database_transferred,
    textmining, textmining_transferred,
    initial_combined) = l


    ## divide by 1000

    neighborhood = float(neighborhood) / 1000
    neighborhood_transferred = float(neighborhood_transferred) / 1000
    fusion = float(fusion) / 1000
    cooccurrence =  float(cooccurrence) / 1000
    homology = float(homology) / 1000
    coexpression = float(coexpression) / 1000
    coexpression_transferred = float(coexpression_transferred) / 1000
    experiments = float(experiments) / 1000
    experiments_transferred = float(experiments_transferred) / 1000
    database = float(database) / 1000
    database_transferred = float(database_transferred) / 1000
    textmining = float(textmining) / 1000
    textmining_transferred = float(textmining_transferred) / 1000
    initial_combined = int(initial_combined)


    ## compute prior away

    neighborhood_prior_corrected                 = compute_prior_away (neighborhood, prior)             
    neighborhood_transferred_prior_corrected     = compute_prior_away (neighborhood_transferred, prior) 
    fusion_prior_corrected                       = compute_prior_away (fusion, prior)             
    cooccurrence_prior_corrected                 = compute_prior_away (cooccurrence, prior)           
    coexpression_prior_corrected                 = compute_prior_away (coexpression, prior)            
    coexpression_transferred_prior_corrected     = compute_prior_away (coexpression_transferred, prior) 
    experiments_prior_corrected                  = compute_prior_away (experiments, prior)   
    experiments_transferred_prior_corrected      = compute_prior_away (experiments_transferred, prior) 
    database_prior_corrected                     = compute_prior_away (database, prior)      
    database_transferred_prior_corrected         = compute_prior_away (database_transferred, prior)
    textmining_prior_corrected                   = compute_prior_away (textmining, prior)            
    textmining_transferred_prior_corrected       = compute_prior_away (textmining_transferred, prior) 

    ## then, combine the direct and transferred scores for each category:

    neighborhood_both_prior_corrected = 1.0 - (1.0 - neighborhood_prior_corrected) * (1.0 - neighborhood_transferred_prior_corrected)
    coexpression_both_prior_corrected = 1.0 - (1.0 - coexpression_prior_corrected) * (1.0 - coexpression_transferred_prior_corrected)
    experiments_both_prior_corrected = 1.0 - (1.0 - experiments_prior_corrected) * (1.0 - experiments_transferred_prior_corrected)
    database_both_prior_corrected = 1.0 - (1.0 - database_prior_corrected) * (1.0 - database_transferred_prior_corrected)
    textmining_both_prior_corrected = 1.0 - (1.0 - textmining_prior_corrected) * (1.0 - textmining_transferred_prior_corrected)

    # Aggregate into dictionary for combination
    toCombine: dict[float] = {"neighborhood": neighborhood_both_prior_corrected, "fusion": fusion_prior_corrected, "cooccurence": cooccurrence_prior_corrected,
                                "coexpression": coexpression_both_prior_corrected, "experiments": experiments_both_prior_corrected, "database": database_both_prior_corrected,
                                "textmining": textmining_both_prior_corrected}
    
    # If there are desired elements, remove everything else
    if(len(desired_methods)>0):
        for method in list(toCombine.keys()):
            if(method not in desired_methods):
                del toCombine[method]
    else:
        # If there is no specific desired elements to refine to, remove undesired elements from the dictionary
        for method in banned_methods:
            if(method in toCombine.keys()):
                del toCombine[method]
            else:
                print(f"Unrecognised method: {method}. Valid Methods:\n{toCombine.keys()}")

    ## next, do the 1 - multiplication:

    combined_score_one_minus = np.prod([1.0 - toCombine[method] for method in toCombine])

    ## and lastly, do the 1 - conversion again, and put back the prior *exactly once*

    combined_score = (1.0 - combined_score_one_minus)            ## 1- conversion
    combined_score *= (1.0 - prior)                              ## scale down
    combined_score += prior                                      ## and add prior.

    ## round

    combined_score = int(combined_score * 1000)
    with open(outputFile, "a+") as f:
        f.write(f"{protein1}\t{protein2}\t{combined_score}\n")

if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description = "STRING Calculation Parser")
    parser.add_argument("--sourceFile",     action = "store", dest = "sourceFile",      default = INPUT_FILE)
    parser.add_argument("--destination",    action = "store", dest = "destinationFile", default = OUTPUT_FILE)
    parser.add_argument("--banned_methods", action = "store", dest = "banned_methods",  default = "textmining")
    parser.add_argument("--desired_methods",action = "store", dest = "desired_methods", default = "")
    parser.add_argument("--logfile",        action = "store", dest = "logfileLoc",      default = "logfileSTRING.txt")
    main(parser)