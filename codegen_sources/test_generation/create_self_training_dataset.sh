# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##!/bin/bash
# Script to create the dataset for self-training

set -e

JAVA_FUNC_DATASET=$1
PATH_TO_TRANSLATION_MODELS=$2
ST_OUTPUT_ROOT_DIR=$3
RUN_SCRIPT_LOCALLY=True
RERUN_ALL_CHUNKS=False
REPO_ROOT="."
echo "Repository root: $REPO_ROOT"

SELECTED_FUNCS_ROOT="${ST_OUTPUT_ROOT_DIR}/selected_functions"
SELECTED_FUNCS="${SELECTED_FUNCS_ROOT}/deduped"
TESTS_PATHS="${ST_OUTPUT_ROOT_DIR}/tests"
TRANSLATIONS_AND_RESULTS_PATH="${ST_OUTPUT_ROOT_DIR}/results"
TESTS_RESULTS="${TRANSLATIONS_AND_RESULTS_PATH}/test_results"
BPE_CODES_PATH="$REPO_ROOT/data/bpe/cpp-java-python/codes"
BPE_VOCAB_PATH="$REPO_ROOT/data/bpe/cpp-java-python/vocab"
PYTHON_BEST_MODEL="${PATH_TO_TRANSLATION_MODELS}/Online_ST_Java_Python.pth"
CPP_BEST_MODEL="${PATH_TO_TRANSLATION_MODELS}/Online_ST_Java_CPP.pth"
SELECTED_TESTS_PATH="${TESTS_PATHS}/selected_tests.csv"
OFFLINE_DATASET_PATH="${ST_OUTPUT_ROOT_DIR}/offline_dataset"
ONLINE_OUTPUT_PATH="${ST_OUTPUT_ROOT_DIR}/online_ST_files"


# Select functions to create tests on
echo "python codegen_sources/test_generation/select_java_inputs.py --local $RUN_SCRIPT_LOCALLY --input_path $JAVA_FUNC_DATASET --output_path $SELECTED_FUNCS_ROOT --rerun $RERUN_ALL_CHUNKS"
python codegen_sources/test_generation/select_java_inputs.py --local $RUN_SCRIPT_LOCALLY --input_path $JAVA_FUNC_DATASET --output_path $SELECTED_FUNCS_ROOT --rerun $RERUN_ALL_CHUNKS

# Create the tests
echo "python codegen_sources/test_generation/create_tests.py --local $RUN_SCRIPT_LOCALLY --input_path $SELECTED_FUNCS --output_path $TESTS_PATHS --rerun $RERUN_ALL_CHUNKS"
python codegen_sources/test_generation/create_tests.py --local $RUN_SCRIPT_LOCALLY --input_path $SELECTED_FUNCS --output_path $TESTS_PATHS --rerun $RERUN_ALL_CHUNKS

# Compute translations and tests results for C++
echo "python codegen_sources/test_generation/compute_transcoder_translations.py --functions_path $SELECTED_FUNCS --csv_path $SELECTED_TESTS_PATH --output_folder $TRANSLATIONS_AND_RESULTS_PATH --bpe_path $BPE_CODES_PATH --target_language python --model_path $PYTHON_BEST_MODEL --local $RUN_SCRIPT_LOCALLY --rerun $RERUN_ALL_CHUNKS"
python codegen_sources/test_generation/compute_transcoder_translations.py \
  --functions_path $SELECTED_FUNCS \
  --csv_path $SELECTED_TESTS_PATH \
  --output_folder $TRANSLATIONS_AND_RESULTS_PATH \
  --bpe_path $BPE_CODES_PATH \
  --target_language cpp \
  --model_path $CPP_BEST_MODEL \
  --local $RUN_SCRIPT_LOCALLY \
  --rerun $RERUN_ALL_CHUNKS
  
# Compute translations and tests results for Python
echo "python codegen_sources/test_generation/compute_transcoder_translations.py --functions_path $SELECTED_FUNCS --csv_path $SELECTED_TESTS_PATH --output_folder $TRANSLATIONS_AND_RESULTS_PATH --bpe_path $BPE_CODES_PATH --target_language python --model_path $PYTHON_BEST_MODEL --local $RUN_SCRIPT_LOCALLY --rerun $RERUN_ALL_CHUNKS"
python codegen_sources/test_generation/compute_transcoder_translations.py \
  --functions_path $SELECTED_FUNCS \
  --csv_path $SELECTED_TESTS_PATH \
  --output_folder $TRANSLATIONS_AND_RESULTS_PATH \
  --bpe_path $BPE_CODES_PATH \
  --target_language python \
  --model_path $PYTHON_BEST_MODEL \
  --local $RUN_SCRIPT_LOCALLY \
  --rerun $RERUN_ALL_CHUNKS

# Select tests and create dataset for offline training
echo "python codegen_sources/test_generation/select_successful_tests.py --input_df $TESTS_RESULTS --output_folder $OFFLINE_DATASET_PATH --bpe_path $BPE_CODES_PATH --bpe_vocab $BPE_VOCAB_PATH"
python codegen_sources/test_generation/select_successful_tests.py \
  --input_df $TESTS_RESULTS \
  --output_folder $OFFLINE_DATASET_PATH \
  --bpe_path $BPE_CODES_PATH \
  --bpe_vocab $BPE_VOCAB_PATH
