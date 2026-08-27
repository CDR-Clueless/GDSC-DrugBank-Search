#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2 Jul 2026

@author: jds40
"""

import os
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from target_functions import get_drugTargets

MANUAL_TARGETS: str = os.path.join("Data", "Derived-Data", "manual_targets.tsv")

def main():
    outputDir = os.path.join("Data", "Results", "Target-Analysis")
    #dTPearson, scPearson = prepare_target_frame()
    #dTGLS, scGLS = prepare_target_frame(os.path.join("Data", "Results", "Survivability-Correlations", "pIC50-GLS_2-AllDrugsByAllGenes.tsv"))
    #target_SC_analysis(saveOutput=outputDir, drugTargets = dT, scScores = sc)
    #get_zScores(outputDir, dT)
    #get_zScores()
    plot_knownDrugs(csvSave = "drug_target_overview.tsv")
    #plot_realScores(drugTargets = dTPearson, scScores = scPearson, saveOutput=None, calcMethod = "Pearson")
    #get_zScores(drugTargets = dTPearson, saveOutput=outputDir, calcMethod = "Pearson", save_stats = True)
    #get_zScores(drugTargets = dTGLS, saveOutput=outputDir, calcMethod = "2-Component GLS", save_stats = True)
    #target_SC_analysis(saveOutput=outputDir, drugTargets=dTGLS, scScores = scGLS, calcMethod = "2-Component GLS")

def get_zScores(saveOutput: Optional[str] = None, drugTargets: Optional[pd.DataFrame] = None,
                titleBase: Optional[str] = None, calcMethod: str = "Pearson",
                save_stats: bool = False) -> None:
    # Change save_stats if no save output was given
    if(saveOutput is None and save_stats == True):
        print(f"A save directory is required to save analysis results")
        save_stats = False
    # Get all known putatitve drug targets
    if(drugTargets is None):
        drugTargets, _ = prepare_target_frame()

    drugTargets = add_thresholds(drugTargets)
    
    zScores = drugTargets["ZSCORE"][~np.isnan(drugTargets["ZSCORE"])]

    plt.scatter(range(zScores.shape[0]), sorted(zScores)[::-1])
    # Add threshold AND p < 0.05 lines (Z-Score of 3 means 3 SD's above norm which is the threshold, Z-Score of 1.645 translates as p<0.05)
    plt.plot([0, zScores.shape[0]], [3.0, 3.0], color = "green")
    plt.plot([0, zScores.shape[0]], [1.645, 1.645], color = "red")
    # Add percentages of how many targets are above the two lines
    threePerc = round((zScores[zScores >= 3.0].shape[0] / zScores.shape[0])*100, 1)
    p5Perc = round((zScores[zScores >= 1.645].shape[0] / zScores.shape[0])*100, 1)
    plt.text(threePerc/100 * zScores.shape[0], 3.0+0.2, f"{threePerc}%", color = "green")
    plt.text(p5Perc/100 * zScores.shape[0], 1.645+0.2, f"{p5Perc}%", color = "red")
    plt.xlabel("Putative Drug Target")
    plt.ylabel("SC Z-Score")
    if(titleBase is None):
        title = f"Z-Scores of Putative Drug Target {calcMethod} Survivability Correlations"
    else:
        title = titleBase
    plt.title(title.replace("CALCMETHOD", calcMethod))
    if(saveOutput is None):
        plt.show()
    else:
        plt.savefig(os.path.join(saveOutput, f"{calcMethod} GDSC All Target Z-Scores"))
        plt.clf()
        plt.close()

    ## Plot the best-scoring gene targets instead of all of them
    highest = np.array([np.nanmax(drugTargets.loc[drugTargets["DRUG_STANDARD"] == drug]["ZSCORE"].values) for drug in drugTargets["DRUG_STANDARD"].unique()], dtype = float)
    highest = highest[~np.isnan(highest)]
    highest = np.array(sorted(list(highest))[::-1])
    print(f"Z-scores imply {highest.shape[0]} drugs with GDSC targets")
    plt.scatter(range(highest.shape[0]), highest)
    # Add threshold AND p < 0.05 lines (Z-Score of 3 means 3 SD's above norm which is the threshold, Z-Score of 1.645 translates as p<0.05)
    plt.plot([0, highest.shape[0]], [3.0, 3.0], color = "green")
    plt.plot([0, highest.shape[0]], [1.645, 1.645], color = "red")
    # Add percentages of how many targets are above the two lines
    threePerc = round((highest[highest >= 3.0].shape[0] / highest.shape[0])*100, 1)
    p5Perc = round((highest[highest >= 1.645].shape[0] / highest.shape[0])*100, 1)
    plt.text(threePerc/100 * highest.shape[0], 3.0+0.2, f"{threePerc}%", color = "green")
    plt.text(p5Perc/100 * highest.shape[0], 1.645+0.2, f"{p5Perc}%", color = "red")
    plt.xlabel("Putative Drug Target")
    plt.ylabel("SC Z-Score")
    if(titleBase is None):
        title = f"Highest Z-Scores of Putative Drug Target {calcMethod} Survivability Correlations"
    else:
        title = titleBase
    plt.title(title.replace("CALCMETHOD", calcMethod))
    if(saveOutput is None):
        plt.show()
    else:
        plt.savefig(os.path.join(saveOutput, f"{calcMethod} GDSC Highest Target Z-Scores"))
        plt.clf()
        plt.close()

    # If stats are wanted to be saved, save here
    if(save_stats == True):
        fileDir = os.path.join(saveOutput, f"{calcMethod} drugTargets.tsv")
        if(os.path.exists(fileDir)):
            df = pd.read_csv(fileDir, sep = "\t")
            df = pd.concat([df, drugTargets])
        else:
            df = drugTargets
        df.drop_duplicates(inplace=True)
        df.to_csv(fileDir, sep = "\t", lineterminator="\n", index = False)
        ## Save lists of gene groups predicted and not predicted as targets
        # First, remove rows with NaN values
        dTnoNan = drugTargets.dropna(axis = 0, subset = ["ZSCORE"])
        # Loop over the two major thresholds - 1.645 (p<0.05) and 3.0 (3 Standard Deviations)
        for threshLabel, thresh in zip(["3SD", "p < 0.05"], [3.0, 1.645]):
            # Now get all drugs above or below the thresholds
            pred3z = dTnoNan.loc[dTnoNan["ZSCORE"] >= thresh]
            nonpred3z = dTnoNan.loc[dTnoNan["ZSCORE"] < thresh]
            # Next get all drugs which have at least one gene above the threshold
            highPred3z = []
            for drug in pred3z["DRUG_STANDARD"].unique():
                entry = pred3z.loc[pred3z["DRUG_STANDARD"] == drug]
                rel = entry.loc[entry["ZSCORE"]==max(entry["ZSCORE"].values)]
                if(rel["ZSCORE"].values[0] >= thresh):
                    highPred3z.append(deepcopy((rel["DRUG_STANDARD"].values[0], rel["TARGET"].values[0])))
            # And then use this to infer all drugs which don't have a single gene above the threshold
            nonHighPred3z = [drug for drug in dTnoNan["DRUG_STANDARD"].values if drug not in [e[0] for e in highPred3z]]
            # Format the above into a single list of lists (padding with blank strings)
            maxLen = max([len(pred3z["TARGET"].values), len(nonpred3z["TARGET"].values), len(highPred3z), len(nonHighPred3z)])
            pred3z = list(pred3z["TARGET"].values) + ([""] * (maxLen - len(pred3z["TARGET"].values)))
            nonpred3z = list(nonpred3z["TARGET"].values) + ([""] * (maxLen - len(nonpred3z["TARGET"].values)))
            highPred3z = [e[0] for e in highPred3z] + ([""] * (maxLen - len(highPred3z)))
            nonHighPred3z = nonHighPred3z + ([""] * (maxLen - len(nonHighPred3z)))            
            outFrame = pd.DataFrame(data = list(zip(pred3z, nonpred3z, highPred3z, nonHighPred3z)),
                                    columns = ["Target Genes Predicted", "Target Genes Not Predicted",
                                               "Drugs With >=1 Target Genes Predicted", "Drugs With No Target Genes Predicted"])
            outFrame.to_csv(os.path.join(saveOutput, f"{calcMethod} Threshold Labels {threshLabel}.tsv"),
                            sep = "\t", lineterminator = "\n", index = False)
    return

# Plot drugs which don't have known targets
def plot_knownDrugs(saveOutput: Optional[str] = None, drugTargets: Optional[pd.DataFrame] = None, scScores: Optional[pd.DataFrame] = None,
                       calcMethod: str = "Pearson", csvSave: str = "", merge_manual: bool = True):
    # Get the unrefined target frame
    unrefined, scScores = prepare_target_frame(refine_frame=False)
    # Get total number of drugs and list of standardised drugs
    GDSCdrugs: list = [str(drug).upper().replace(" ","").replace("_", "").replace("(","").replace(")","") for drug in scScores.columns]
    total_drugs = len(GDSCdrugs)
    GDSCgenes: dict = {str(gene).upper().replace(" ","").replace("_", "").replace("(","").replace(")",""): True for gene in scScores.index}
    # Format targets in target frame just to be safe
    unrefined["TARGET"] = unrefined.TARGET.str.upper().replace(" ","").replace("_", "").replace("(","").replace(")","")
    # Refine to just drugs in GDSCdrugs
    unrefined = unrefined.loc[unrefined["DRUG_STANDARD"].isin(GDSCdrugs)]
    # Get drugs for which there are manual targets identified
    manual = pd.read_csv(MANUAL_TARGETS, sep = "\t")
    manual.dropna(axis = "index", how = "any", inplace = True)
    manual = manual.loc[manual["TARGET"]!=""]
    manual["DRUG_STANDARD"] = [str(drug).upper().replace(" ","").replace("_", "").replace("(","").replace(")","") for drug in manual["DRUG"].values]
    manual = manual.loc[manual["DRUG_STANDARD"].isin(GDSCdrugs)]
    manual_drugs = [d for d in manual["DRUG_STANDARD"].unique()]
    # Get drugs for which there are no known targets, known but non-DepMap gene targets, and at least 1 DepMap gene target
    unknown_targets_drugs: list = []
    known_targets_drugs: list = []
    good_targets_drugs: list = []
    for d in unrefined["DRUG_STANDARD"].unique():
        rel = unrefined.loc[unrefined["DRUG_STANDARD"]==d]
        # First check: If all targets are empty, add it to unknown target drugs and continue
        if(not (~(rel["TARGET"]=="")).any()):
            unknown_targets_drugs.append(d)
        # Second check: If there is at least 1 target which is a DepMap gene, add it to good drug targets and continue
        elif((rel["TARGET"].isin(GDSCgenes)).any()):
            good_targets_drugs.append(d)
        # Final possibility: If there are any known targets but they're not in GDSC, add it to known target drugs and continue
        else:
            known_targets_drugs.append(d)
    # Get drugs for which absolutely NO information is available
    unknown_drugs = [d for d in GDSCdrugs if d not in good_targets_drugs + manual_drugs + known_targets_drugs + unknown_targets_drugs]

    # Remove manually-identified drugs from other lists if desired
    if(merge_manual):
        unknown_drugs = [d for d in unknown_drugs if d not in manual_drugs]
        unknown_targets_drugs = [d for d in unknown_targets_drugs if d not in manual_drugs]
        known_targets_drugs = [d for d in known_targets_drugs if d not in manual_drugs]
        good_targets_drugs = [d for d in good_targets_drugs if d not in manual_drugs]
        print(manual_drugs)
        print(manual_drugs[0] in good_targets_drugs)
        print(f"knownDrugs finds:\n{len(manual_drugs)} drugs with manually-identified targets\n{len(good_targets_drugs)} drugs with GDSC targets\
\n{len(known_targets_drugs)} drugs with known non-GDSC targets\
\n{len(unknown_targets_drugs)} drugs with no known targets\n{len(unknown_drugs)} drugs with no information known")

        plt.pie([len(good_targets_drugs), len(manual_drugs), len(known_targets_drugs), len(unknown_targets_drugs), len(unknown_drugs)],
                labels = ["Drugs w DepMap Gene Targets", "Drugs w Manually Identified Targets",
                          "Drugs w Known Targets", "Drugs w/o Known Targets", "Unidentified Drugs"],
                autopct = "%.1f")

    else:
        plt.pie([len(good_targets_drugs), len(known_targets_drugs), len(unknown_targets_drugs), len(unknown_drugs)],
                labels = ["Drugs w DepMap Gene Targets", "Drugs w Known Targets", "Drugs w/o Known Targets", "Unidentified Drugs"],
                autopct = "%.1f")
    if(saveOutput is None):
        plt.show()
    else:
        plt.savefig(os.path.join(saveOutput, "Drug Target Knowledge Comparison.png"))
    plt.clf()
    plt.close()
    if(csvSave != ""):
        # Get lists of good, manual and known target lists
        manual_drugs_targets: list = []
        for d in manual_drugs:
            rel = manual.loc[manual["DRUG_STANDARD"]==d]
            manual_drugs_targets.append(rel["TARGET"].values)
        good_targets_drug_targets: list = []
        for d in good_targets_drugs:
            rel = unrefined.loc[unrefined["DRUG_STANDARD"]==d]
            good_targets_drug_targets.append(rel["TARGET"].values)
        known_targets_drugs_targets: list = []
        for d in known_targets_drugs:
            rel = unrefined.loc[unrefined["DRUG_STANDARD"]==d]
            known_targets_drugs_targets.append(rel["TARGET"].values)
        # Pad lists to all be same length
        maxL = max([len(good_targets_drugs), len(manual_drugs), len(known_targets_drugs), len(unknown_targets_drugs), len(unknown_drugs)])
        for l in [good_targets_drugs, manual_drugs, known_targets_drugs, unknown_targets_drugs, unknown_drugs,
                  manual_drugs_targets, good_targets_drug_targets, known_targets_drugs_targets]:
            l += [""] * (maxL - len(l))
        df = pd.DataFrame(data = list(zip(good_targets_drugs, good_targets_drug_targets,
                                          manual_drugs, manual_drugs_targets,
                                          known_targets_drugs, known_targets_drugs_targets,
                                          unknown_targets_drugs, unknown_drugs)),
                          columns = ["Drugs with at least 1 DepMap Target", "Targets",
                                     "Drugs with Manually Identified Targets", "Targets",
                                     "Drugs with Targets not in DepMap Genes", "Targets",
                                     "Drugs with no Known Targets", "Unknown Drugs"])
        if(csvSave.split(".")[-1].lower()=="tsv"):
            sep = "\t"
        else:
            sep = ","
        df.to_csv(csvSave, sep = sep, lineterminator="\n", index = False)
    return

def plot_realScores(saveOutput: Optional[str] = None, drugTargets: Optional[pd.DataFrame] = None, scScores: Optional[pd.DataFrame] = None,
                       calcMethod: str = "Pearson"):
    if(drugTargets is None or scScores is None):
        drugTargets, scScores = prepare_target_frame()

    # Add threshold and SC Target ratios
    drugTargets = add_thresholds(drugTargets)

    # If save dir is given, save this data
    if(saveOutput is not None):
        fileDir = os.path.join(saveOutput, f"{calcMethod} drugTargets.tsv")
        if(os.path.exists(fileDir)):
            df = pd.read_csv(fileDir, sep = "\t")
            df = pd.concat([df, drugTargets])
        else:
            df = drugTargets
        df.drop_duplicates(inplace=True)
        df.to_csv(fileDir, sep = "\t", lineterminator="\n", index = False)

    # Get relevant columns and drop NaN rows
    rdT = drugTargets[["DRUG_STANDARD", "TARGET", "SURVIVABILITY CORRELATION", "THRESHOLD", "p<0.05"]]
    rdT.dropna(axis = "index", subset = ["SURVIVABILITY CORRELATION", "THRESHOLD", "p<0.05"], inplace=True)
    rdT.sort_values("THRESHOLD", axis = "index", ascending = False, inplace=True)

    # Plot thresholds, p<0.05 values and drug values
    plt.plot(range(len(rdT)), rdT["THRESHOLD"].values, label = "Threshold Value", color = "green")
    plt.plot(range(len(rdT)), rdT["p<0.05"].values, label = "p<0.05 Cutoff", color = "blue")
    plt.scatter(range(len(rdT)), rdT["SURVIVABILITY CORRELATION"].values, color = "orange")
    # Remove xtick labels
    plt.xticks(ticks = range(len(rdT)), labels = ["" for _ in range(len(rdT))])
    # Plot percentage of values above and below threshold
    above, below = len(rdT.loc[rdT["SURVIVABILITY CORRELATION"]>=rdT["THRESHOLD"]])/len(rdT), len(rdT.loc[rdT["SURVIVABILITY CORRELATION"]<rdT["THRESHOLD"]])/len(rdT)
    abovep, belowp = len(rdT.loc[rdT["SURVIVABILITY CORRELATION"]>=rdT["p<0.05"]])/len(rdT), len(rdT.loc[rdT["SURVIVABILITY CORRELATION"]<rdT["p<0.05"]])/len(rdT)
    for ymod, perc, col in zip([lambda x : max(x)-0.05, max], [abovep, above], ["blue", "green"]):
        plt.text(len(rdT)/2, ymod(rdT["SURVIVABILITY CORRELATION"].values), f"{perc*100:.1f}%", color = col)
    plt.xlabel("Drug Target")
    plt.ylabel("Survivability Correlation")
    plt.title(f"{calcMethod} Survivability Correlation Target Values")
    plt.legend(loc = "upper right")

    ## Plot SC values
    if(saveOutput is None):
        plt.show()
    else:
        plt.savefig(os.path.join(saveOutput, f"{calcMethod} GDSC All Target Scores Plus Threshold.png"))
    plt.clf()

def add_thresholds(drugTargets: pd.DataFrame):
    drugTargets["THRESHOLD"] = drugTargets["DRUG_MEAN"] + (3*drugTargets["DRUG_SD"])
    drugTargets["SURVIVABILITY TARGET RATIO"] = drugTargets["SURVIVABILITY CORRELATION"] / drugTargets["THRESHOLD"]
    drugTargets["ZSCORE"] = np.divide(drugTargets["SURVIVABILITY CORRELATION"] - drugTargets["DRUG_MEAN"], drugTargets["DRUG_SD"])
    drugTargets["p<0.05"] = (1.645 * drugTargets["DRUG_SD"]) + drugTargets["DRUG_MEAN"]
    return drugTargets

def target_SC_analysis(saveOutput: Optional[str] = None, drugTargets: Optional[pd.DataFrame] = None, scScores: Optional[pd.DataFrame] = None,
                       calcMethod: str = "Pearson") -> None:
    if(drugTargets is None or scScores is None):
        drugTargets, scScores = prepare_target_frame()

    # Add threshold and SC Target ratios
    drugTargets = add_thresholds(drugTargets)

    # If save dir is given, save this data
    if(saveOutput is not None):
        fileDir = os.path.join(saveOutput, f"{calcMethod} drugTargets.tsv")
        if(os.path.exists(fileDir)):
            df = pd.read_csv(fileDir, sep = "\t")
            df = pd.concat([df, drugTargets])
        else:
            df = drugTargets
        df.drop_duplicates(inplace=True)
        df.to_csv(fileDir, sep = "\t", lineterminator="\n", index = False)

    realRatios = drugTargets["SURVIVABILITY TARGET RATIO"][~np.isnan(drugTargets["SURVIVABILITY TARGET RATIO"])]
    realVals = drugTargets["SURVIVABILITY CORRELATION"][~np.isnan(drugTargets["SURVIVABILITY CORRELATION"])]
    print(f"{realVals.shape[0]} SC values found out of {len(drugTargets)} rows")

    ## Plot SC values
    ys = sorted(realVals)[::-1]
    plt.scatter(range(realVals.shape[0]), ys, color = "b")
    plt.xlabel("Drug Target")
    plt.ylabel("Target Survivability Correlation")
    plt.title("Survivability Correlation of Putative Drug Targets")
    if(saveOutput is None):
        plt.show()
    else:
        plt.savefig(os.path.join(saveOutput, f"{calcMethod} GDSC All Target Scores.png"))
    plt.clf()

    ## Plot best SC scores for each drug
    ys = np.array([np.nanmax(drugTargets.loc[drugTargets["DRUG_STANDARD"] == drug]["SURVIVABILITY CORRELATION"].values) for drug in drugTargets["DRUG_STANDARD"].unique()], dtype = float)
    ys = np.array(sorted(ys[~np.isnan(ys)])[::-1])
    plt.scatter(range(ys.shape[0]), ys, color = "b")
    plt.xlabel("Drug Target")
    plt.ylabel("Target Survivability Correlation")
    plt.title("Survivability Correlation of Highest Scoring Putative Drug Targets")
    if(saveOutput is None):
        plt.show()
    else:
        plt.savefig(os.path.join(saveOutput, f"{calcMethod} GDSC Best Target Scores.png"))
    plt.clf()

    ## Plot SC Ratios
    ys = sorted(realRatios)[::-1]
    for i in range(len(ys)-1):
        if(ys[i]>=1 and ys[i+1] < 1):
            cutoff = float(i)+0.5
            cutoffPerc = (i+1)/len(ys)
    plt.scatter(range(realRatios.shape[0]), ys, color = "b")
    plt.plot([0, realRatios.shape[0]], [1, 1], linestyle = "--", color = "red")
    # Plot cutoff point and text
    plt.plot([cutoff, cutoff], [ys[-1], ys[0]], linestyle = "--", color = "g")
    plt.text(cutoff, ys[0], f"{cutoffPerc*100:.2f}%", color = "g")
    plt.xlabel("Drug Target")
    plt.ylabel("Target Correlation Ratio")
    plt.title("SC Score-SC Threshold Ratios of Putative Drug Targets")
    if(saveOutput is None):
        plt.show()
    else:
        plt.savefig(os.path.join(saveOutput, f"{calcMethod} GDSC All Target Scores ratios.png"))
    plt.clf()

    ## Plot the highest ratio for each drug
    maxRatios = []
    for drug in drugTargets["DRUG"].unique():
        ratios = drugTargets.loc[drugTargets["DRUG"]==drug]["SURVIVABILITY TARGET RATIO"].values
        # Remove NaN values
        ratios = ratios[~np.isnan(ratios)]
        if(ratios.shape[0]>0):
            maxRatios.append(np.max(ratios))
    ys = sorted(maxRatios)[::-1]
    for i in range(len(ys)-1):
        if(ys[i]>=1 and ys[i+1] < 1):
            cutoff = float(i)+0.5
            cutoffPerc = (i+1)/len(ys)
    plt.scatter(range(len(maxRatios)), ys, color = "b")
    plt.plot([0, len(maxRatios)-1], [1, 1], linestyle = "--", color = "red")
    # Plot cutoff point and text
    plt.plot([cutoff, cutoff], [ys[-1], ys[0]], linestyle = "--", color = "g")
    plt.text(cutoff, ys[0], f"{cutoffPerc*100:.2f}%", color = "g")
    plt.xlabel("Best Scoring Drug Target")
    plt.ylabel("Target Correlation Ratio")
    plt.title("SC Score-SC Threshold Ratios of Best Scoring Putative Drug Targets")
    if(saveOutput is None):
        plt.show()
    else:
        plt.savefig(os.path.join(saveOutput, f"{calcMethod} GDSC Best Target Scores ratios.png"))
    plt.clf()

    ## Plotting number of genes with SC scores above putative targets
    # Dictionary for storing {drug: Target gene: Number of genes above target gene score}
    results = {}
    for drug, gene in zip(drugTargets["DRUG_STANDARD"].values, drugTargets["TARGET"].values):
        if(drug not in results):
            results[drug] = {}
        if(drug not in scScores.columns):
            results[drug][gene] = np.nan
            continue
        elif(gene not in scScores.index):
            results[drug][gene] = np.nan
            continue
        val = scScores[drug].loc[gene]
        if(type(val)!=np.float64):
            # There seems to be some weird issue with some genes being duplicated in the Survivability Correlations index,
            # so I'm just using the maximum value found between these two for now
            val = np.nanmax(val.values)
        results[drug][gene] = np.sum((scScores[drug].values>val))
    
    counts = np.array([results[drug][gene] for drug in results for gene in results[drug]], dtype = float)
    counts = sorted(counts[~np.isnan(counts)].astype(int))
    plt.scatter(range(len(counts)), counts)
    plt.xlabel("Drug Target")
    plt.ylabel("Number of higher-SC-scoring genes")
    plt.title("Number of Genes with Higher SC values than Putative Target Genes")
    if(saveOutput is None):
        plt.show()
    else:
        plt.savefig(os.path.join(saveOutput, f"{calcMethod} GDSC All Higher Target SC Counts.png"))
    plt.clf()

    # Plot these numbers but for the best-performing target per drug
    bestCounts: np.ndarray = np.zeros(len(results), dtype = float)
    for i, drug in enumerate(results.keys()):
        bestCounts[i] = np.nanmax([results[drug][gene] for gene in results[drug]])
    bestCounts = sorted(bestCounts[~np.isnan(bestCounts)].astype(int))
    plt.scatter(range(len(bestCounts)), bestCounts)
    plt.xlabel("Best Drug Target")
    plt.ylabel("Number of higher-SC-scoring genes")
    plt.title("Number of Genes with Higher SC values than Best Drug Putatitve Target Gene")
    if(saveOutput is None):
        plt.show()
    else:
        plt.savefig(os.path.join(saveOutput, f"{calcMethod} GDSC Best Higher Target SC Counts.png"))
    plt.clf()

    ## Get details on drugs missing from putatitve target lists and putative targets missing from GDSC Survivability Correlation data
    drugs_missing, gene_missing, extra_drugs, nonTarget_drugs = {}, {}, {}, {}
    for drug, gene in zip(drugTargets["DRUG_STANDARD"].values, drugTargets["TARGET"].values):
        if(drug not in scScores.columns):
            drugs_missing[drug] = True
        elif(gene not in scScores.index):
            gene_missing[gene] = True
    for drug in scScores.columns:
        if(drug not in drugTargets["DRUG_STANDARD"].values):
            extra_drugs[drug] = True
        else:
            someGene = False
            for gene in drugTargets.loc[drugTargets["DRUG_STANDARD"]==drug]["TARGET"].values:
                if(gene in scScores.index):
                    someGene = True
            if(not someGene):
                nonTarget_drugs[drug] = True

    drugMissingString: str = ', '.join([str(drug) for drug in drugs_missing.keys()])
    geneMissingString: str = ', '.join([str(gene) for gene in gene_missing.keys()])
    extraDrugString: str = ', '.join([str(drug) for drug in extra_drugs.keys()])
    nonTargetDrugString: str = ', '.join([str(drug) for drug in nonTarget_drugs.keys()])
    
    outString: str = f"{len(drugs_missing)} Drugs in putative target list not found in GDSC Data:\n{drugMissingString}\n\n\
{len(gene_missing)} Gene Targets in putative target list not found in GDSC Data:\n{geneMissingString}\n\n\
{len(extra_drugs)} Drugs in GDSC Data not found in putative target list:\n{extraDrugString}\n\n\
{len(nonTarget_drugs)} Drugs in GDSC Data with no valid putative target:\n{nonTargetDrugString}"
    if(saveOutput is None):
        print(outString)
    else:
        with open(os.path.join(saveOutput, f"{calcMethod} Invalid Drugs and Targets.txt"), "w") as f:
            f.write(outString)

    return

def prepare_target_frame(scFrameLoc: str = os.path.join("Data", "Results", "Survivability-Correlations", "pIC50-AllDrugsByAllGenes.tsv"),
                         refine_frame: bool = True) -> Tuple[pd.DataFrame,pd.DataFrame]:
    # Get all known putatitve drug targets
    drugTargets = get_drugTargets(include_manual=refine_frame)

    ## Get SC ratio scores for each target
    scScores = pd.read_csv(scFrameLoc, sep = "\t")
    scScores.set_index("symbol", inplace=True)
    # Format columns/values on each dataframe
    scScores.columns = [str(col).upper().replace(" ","").replace("_", "").replace("(","").replace(")","") for col in scScores.columns]
    drugTargets["DRUG_STANDARD"] = [str(drug).upper().replace(" ","").replace("_", "").replace("(","").replace(")","") for drug in drugTargets["DRUG"].values]
    # List for storing tuples of (drug, Target gene SC score, drug mean, drug SD)
    results = []
    for drug, gene in zip(drugTargets["DRUG_STANDARD"].values, drugTargets["TARGET"].values):
        if(drug not in scScores.columns):
            results.append((drug, np.nan, np.nan, np.nan))
            continue
        elif(gene not in scScores.index):
            results.append((drug, np.nan, np.nanmean(scScores[drug].values), np.nanstd(scScores[drug].values)))
            continue
        val = scScores[drug].loc[gene]
        if(type(val)==np.float64):
            results.append((drug, val, np.nanmean(scScores[drug].values), np.nanstd(scScores[drug].values)))
        else:
            # There seems to be some weird issue with some genes being duplicated in the Survivability Correlations index,
            # so I'm just using the maximum value found between these two for now
            results.append((drug, max(val.values), np.nanmean(scScores[drug].values), np.nanstd(scScores[drug].values)))
    
    relSC = np.array([result[1] for result in results], dtype = float)
    means = np.array([result[2] for result in results], dtype = float)
    sds = np.array([result[3] for result in results], dtype = float)

    drugTargets["SURVIVABILITY CORRELATION"] = relSC
    drugTargets["DRUG_MEAN"] = means
    drugTargets["DRUG_SD"] = sds

    if(refine_frame):
        return drugTargets[drugTargets.TARGET != ""], scScores
    return drugTargets, scScores

if(__name__=="__main__"):
    main()