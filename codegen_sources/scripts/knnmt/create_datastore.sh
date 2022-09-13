#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=create_datastore
#SBATCH --output=transcoder_st_create_datastore_%j.log

python -m codegen_sources.scripts.knnmt.datastore