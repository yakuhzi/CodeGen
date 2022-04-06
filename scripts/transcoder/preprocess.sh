#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=10GB
#SBATCH --partition multiple
#SBATCH --gres=gpu:0
#SBATCH --job-name=train_data_mlm_transcoder
#SBATCH --output=train_data_mlm_transcoder_%j.out

DATASET_PATH='dataset/Python'
TEMP_DIR='dataset/Python/temp'
NGPU=1

for FILE in $DATASET_PATH/*.gz
do
  mkdir $TEMP_DIR
  cp $FILE $TEMP_DIR

  python -m codegen_sources.preprocessing.preprocess \
      --langs cpp java python \
      --mode=monolingual_functions \
      --local=True \
      --bpe_mode=fast \
      --train_splits="$NGPU" \
      "$TEMP_DIR"

  for OUT_FILE in $TEMP_DIR/*.tok
  do
    mv $OUT_FILE $DATASET_PATH
  done

  rm -r $TEMP_DIR
done
