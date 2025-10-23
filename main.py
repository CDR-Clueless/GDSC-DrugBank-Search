import os
import numpy as np
import pandas as pd

def main():
    db, g1, g2 = get_data()
    if(db==None or g1==None or g2==None):
        return
    return

def get_data():
    db, g1, g2 = None, None, None
    for filename in os.listdir("Data"):
        if("gdsc1" in filename.lower()):
            g1 = pd.read_excel(os.path.join("Data",filename))
        elif("gdsc2" in filename.lower()):
            g2 = pd.read_excel(os.path.join("Data",filename))
        elif(filename.lower()=="full database.xml"):
            db = os.path.join("Data",filename)
    if(db==None or g1==None or g2==None):
        errors = ""
        if(db==None):
            errors += "DrugBank, "
        if(g1==None):
            errors += "GDSC1, "
        if(g2==None):
            errors += "GDSC2, "
        errors = errors[:-2]
        print(f"Error: cannot find file for {errors}")
        return None, None, None
    return db, g1, g2


if __name__ == "__main__":
    main()
