#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=10GB
#SBATCH --partition multiple
#SBATCH --gres=gpu:0
#SBATCH --job-name=preprocess_java_functions
#SBATCH --output=preprocess_java_functions_%j.log

DATASET_PATH=data/java_functions
NGPU=1

python -m codegen_sources.preprocessing.preprocess \
    --langs java \
    --mode=monolingual_functions \
    --local=False \
    --bpe_mode=fast \
    --train_splits="$NGPU" \
    "$DATASET_PATH"