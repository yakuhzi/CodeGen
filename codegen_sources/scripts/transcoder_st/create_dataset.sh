#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem=150GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=create_dataset
#SBATCH --output=transcoder_st_create_dataset_%j.log

JAVA_FUNC_DATASET='data/java_functions'
MODELS_PATH='models/transcoder_st'
OUTPUT_DIRECTORY='data/parallel_corpus'

# Create data (it will take a while)
bash codegen_sources/test_generation/create_self_training_dataset.sh $JAVA_FUNC_DATASET $MODELS_PATH $OUTPUT_DIRECTORY