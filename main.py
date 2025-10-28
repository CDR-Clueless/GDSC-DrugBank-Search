import os
import numpy as np
import pandas as pd
from lxml import etree

from typing import Optional

DRUG_TAG_PREFIX: str = "{http://www.drugbank.ca}"

def main():
    db, g1, g2 = get_data()
    if(db is None or g1 is None or g2 is None):
        return
    # The root tag is '{http://www.drugbank.ca}drugbank', while each subitem has the '{http://www.drugbank.ca}drug' tag
    root = db.getroot()

    runique = pd.Series(r.tag for r in root).unique()
    print(f"Unique items in root:\n{runique}")
    seunique = pd.Series(se.tag for se in root[0]).unique()
    print(f"Unique tags in the root tree:\n{seunique}")

    for element in root[0].iter(DRUG_TAG_PREFIX+"pharmacodynamics"):
        print(etree.tostring(element))
        eunique = pd.Series(e.tag for e in element).unique()
        print(f"Unique items in top element:\n{eunique}")
        for i in element.iterchildren():
            print("Unique items in first child of top element:")
            print(pd.Series([ie.tag for ie in i]).unique())
            break
    return

def get_data() -> tuple[Optional[etree.ElementTree], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    db, g1, g2 = None, None, None
    if(os.path.exists("Data")==False):
        os.mkdir("Data")
        os.mkdir(os.path.join("Data","Raw Data"))
    for filename in os.listdir(os.path.join("Data", "Raw Data")):
        if("gdsc1" in filename.lower()):
            g1 = pd.read_excel(os.path.join("Data", "Raw Data", filename))
            print(f"Imported GDSC1 from {filename}")
        elif("gdsc2" in filename.lower()):
            g2 = pd.read_excel(os.path.join("Data", "Raw Data", filename))
            print(f"Imported GDSC2 from {filename}")
        elif(filename.lower()=="full database.xml"):
            with open(os.path.join("Data", "Raw Data", filename), "r") as f:
                db = etree.parse(f)
            print(f"Imported DrugBank from {filename}")
    if(db is None or g1 is None or g2 is None):
        errors = ""
        if(db is None):
            errors += "DrugBank, "
        if(g1 is None):
            errors += "GDSC1, "
        if(g2 is None):
            errors += "GDSC2, "
        errors = errors[:-2]
        print(f"Error: cannot find file for {errors}")
        return None, None, None
    return db, g1, g2


if __name__ == "__main__":
    main()
