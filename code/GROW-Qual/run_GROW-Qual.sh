#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -p rome
#SBATCH -N 1
#SBATCH -n 128

cd /gpfs/work4/0/dynql/GROW-Qual/

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /gpfs/home6/ejones/.conda/envs/myenv

### ---------------------------------------- ###
###               PROCESS CSVS               ###
### ---------------------------------------- ###
#python scripts/GROW-Qual/create_GROW-Qual_csvs.py

### ---------------------------------------- ###
###               PROCESS ZARR               ###
### ---------------------------------------- ###
#python scripts/GROW-Qual/create_GROW-Qual_zarr.py

### ---------------------------------------- ###
###      DERIVE EARTH SYSTEM VARIABLES       ###
### ---------------------------------------- ###
#bash scripts/GROW-Qual/submit_esv_jobs.sh

### ---------------------------------------- ###
###     INTEGRATE EARTH SYSTEM VARIABLES     ###
### ---------------------------------------- ###
python scripts/GROW-Qual/integrate_esv_into_GROW-Qual.py --all