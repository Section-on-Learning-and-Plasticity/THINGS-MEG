# THINGS-MEG

To run the analysis and generate the plots, the BIDS formatted THINGS-MEG data needs to be downloaded from OpenNeuro (https://openneuro.org/datasets/ds004212/versions/2.0.0). Version 2.0.0 contains all MEG data but does not contain the necessary folders within source data that are needed to run all the analyses (e.g., the aggregate dimension weights for all images are missing). A new version of the data will be made available once the paper is published.

All analysis scripts (analysis*.py) are command-line-callable. The correct path location to the BIDS-data has to be provided. The figures are generate from the "figure*.ipynb" notebooks. Within the notebooks the path has to be set to the location of the bids data.
