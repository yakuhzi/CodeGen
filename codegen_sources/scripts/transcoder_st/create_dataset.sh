#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:0
#SBATCH --job-name=transcoder_st_create_dataset
#SBATCH --output=transcoder_st_create_dataset_%j.log

JAVA_FUNC_DATASET=$(ws_find code-gen)/dataset/java
MODELS_PATH='models/transcoder_st'
OUTPUT_DIRECTORY='dump/transcoder_st/dataset'

# Create data (it will take a while)
bash codegen_sources/test_generation/create_self_training_dataset.sh $JAVA_FUNC_DATASET $MODELS_PATH $OUTPUT_DIRECTORY