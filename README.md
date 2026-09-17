# GROW-Qual

*GROW-Qual* is an open access dataset that brings groundwater quality to the research paradigm of large sample hydrology (LSH), integrating daily groundwater quality data from 92 constituents with catchment attributes, meteorological forcing and co-located groundwater depth/level observations.

The full dataset can be accessed [here](https://doi.org/10.5281/zenodo.22638888). This contains both the full dataset in .Zarr format in addition to all processed water quality observations in .csv format.

A manuscript associated with the dataset is currently under development.

## About this repository
This repository contains the code for recreating or extending the *GROW-Qual* dataset.

The code/ folder is split across three sub-folders:
* add_data/: scripts for adding groundwater quality data and deriving linkages to GGMN and GROW groundwater quantity monitoring stations.  
* GROW-Qual/: scripts for processing water quality and quantity data to create the *GROW-Qual* dataset, both as .csv's and in .zarr format.

A full list of variables included in *GROW-Qual* is also provided in this repository (GROW-Qual_zarr_variables.csv).

## *GROW*
*GROW-Qual* adds groundwater quality observations to the existing *GROW* [dataset](https://doi.org/10.5281/zenodo.15149480). The original *GROW* paper can be accessed [here](https://www.nature.com/articles/s41597-026-06966-1), while developments are documented on GROWS's [GitHub](https://github.com/EarthSystemModelling/GROW/) page. This includes the methodology applied for deriving contextual environmental data (transient and static) variables in *GROW-Qual*, which directly follows the approach developed for *GROW*. 
