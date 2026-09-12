#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p rome
#SBATCH -N 1
#SBATCH -n 128

cd /gpfs/work4/0/dynql/GROW-Qual/

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /gpfs/home6/ejones/.conda/envs/myenv

### ---------------------------------------- ###
###       PROCESS INDIVIDUAL DATASETS        ###
### ---------------------------------------- ###

###Add data
python scripts/add_data/add_gwq_dataset.py GGMN
python scripts/add_data/add_gwq_dataset.py GEMS
python scripts/add_data/add_gwq_dataset.py Waterbase
python scripts/add_data/add_gwq_dataset.py WQP
python scripts/add_data/add_gwq_dataset.py EMPODAT
python scripts/add_data/add_gwq_dataset.py UK_EA
python scripts/add_data/add_gwq_dataset.py Aus_BOM
python scripts/add_data/add_gwq_dataset.py NWDP

### ---------------------------------------- ###
###            MATCH GWL to GWQ              ###
### ---------------------------------------- ###
python scripts/add_data/add_ggmn_grow_id.py