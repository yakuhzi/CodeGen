#!/bin/bash

DATASET_PATH='dataset/CPP'
TEMP_DIR="${DATASET_PATH}/temp"
MODE='monolingual_functions'
NGPU=1

for FILE in "$DATASET_PATH"/*.gz
do
  mkdir "$TEMP_DIR"
  cp "$FILE" "$TEMP_DIR"

  python -m codegen_sources.preprocessing.preprocess \
      --langs cpp java python \
      --mode="$MODE" \
      --local=True \
      --bpe_mode=fast \
      --train_splits="$NGPU" \
      "$TEMP_DIR"

  for OUT_FILE in "$TEMP_DIR"/*.tok
  do
    zip "$OUT_FILE".zip "$OUT_FILE"
  done

  for ZIP_FILE in "$TEMP_DIR"/*.zip
  do
    mv "$ZIP_FILE" "$DATASET_PATH"
  done

  rm -r "$TEMP_DIR"
done
