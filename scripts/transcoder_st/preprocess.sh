#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=10GB
#SBATCH --partition multiple
#SBATCH --gres=gpu:0
#SBATCH --job-name=train_data_mlm_transcoder
#SBATCH --output=train_data_mlm_transcoder_%j.out

DATASET_PATH=$(ws_find code-gen)/dataset/single
NGPU=1

python -m codegen_sources.preprocessing.preprocess \
    --langs java \
    --mode=monolingual_functions \
    --local=False \
    --bpe_mode=fast \
    --train_splits="$NGPU" \
    "$DATASET_PATH"