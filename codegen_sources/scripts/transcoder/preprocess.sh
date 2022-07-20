#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:0
#SBATCH --job-name=preprocess
#SBATCH --output=transcoder_preprocess_%j.log

DATASET_PATH=$(ws_find code-gen)/dataset/all
MODE='monolingual_functions'
NGPU=1

python -m codegen_sources.preprocessing.preprocess \
    --langs cpp java python \
    --mode="$MODE" \
    --local=True \
    --bpe_mode=fast \
    --train_splits="$NGPU" \
    "$DATASET_PATH"
