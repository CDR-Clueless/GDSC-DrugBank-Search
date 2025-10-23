# GDSC-DrugBank-Search
Tool to find drugs tested within GDSC dataset, locate these drugs in the DrugBank dataset, and find genes associated with this drug in the DrugBank dataset

Setup:
    Download the datasets 'All drugs' from DrugBank (https://go.drugbank.com/releases/latest#full) - this will need a DrugBank academic account.
    Download the datasets 'GDSC1-dataset' and 'GDSC2-dataset' from the CancerRXgene website (https://www.cancerrxgene.org/downloads/bulk_download) - this should be freely available
    Create a folder named 'Data' within the downloaded github repository. Move all downloaded files here, extract the DrugBank data, and delete the zip folder if desired.
    Ensure the GDSC1 data file (e.g. GDSC1_fitted_dose_response_27Oct23.xlsx) has the word 'GDSC1' in its name, ensure the GDSC2 data file has 'GDSC2' in its name, and ensure the DrugBank file is named 'full database.xml'
    The main script can now be run
