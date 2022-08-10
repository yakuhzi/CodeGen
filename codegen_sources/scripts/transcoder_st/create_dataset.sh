#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem=150GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=cdg_0
#SBATCH --output=transcoder_st_create_dataset_0_%j.log

JAVA_FUNC_DATASET=$(ws_find code-gen)/dataset/single/java84
MODELS_PATH='models/transcoder_st'
OUTPUT_DIRECTORY='dump/transcoder_st/dataset_84'

# Create data (it will take a while)
bash codegen_sources/test_generation/create_self_training_dataset.sh $JAVA_FUNC_DATASET $MODELS_PATH $OUTPUT_DIRECTORY