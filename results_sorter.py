import os
import json

def sort_histograms() -> None:
    # Go through the two options: drug-gene and gene-drug and get the appropriate directories
    for option in ["drug", "gene"]:
        mirror = {"drug": "gene", "gene": "drug"}[option]
        dirname = os.path.join("Data", "Results", f"{option.capitalize()}-{mirror} correlation frequency histograms")
        # Check that the histogram directory exists
        if(not os.path.exists(dirname)):
            print(f"No directory found for option {drug} (no directory {dirname}); skipping...")
            continue
        # Make directories for the 3 modalities
        for mod in ["unimodal", "bimodal", "unclear"]:
            os.mkdir(os.path.join(dirname, mod))
        # Get modality (and other) information
        with open(os.path.join(dirname, "stats.json"), "r") as f:
            stats = json.load(f)
        # Iterate over drugs/genes in the stats json/dictionary
        for oname in stats.keys():
            # Get filename for this drug/gene
            fname = ""
            for filename in os.listdir(dirname):
                if(oname in filename):
                    fname = filename
                    break
            if(fname==""):
                print(f"No file found for {option} {oname}; skipping...")
                continue
            # Get modality information
            mod = stats[oname]["modality details"]["modality"]
            moddir = os.path.join(dirname, mod)
            # Move the file to the new directory
            os.replace(os.path.join(dirname, filename), os.path.join(dirname, mod, fname))
    return
