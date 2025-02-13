#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval
#SBATCH --output=eval_transcoder_st_%j.log

JAVA_FUNC_DATASET='dataset/transcoder_st'
MODELS_PATH='models'
OUTPUT_PATH='output/transcoder_st'
DUMP_PATH='dump/transcoder_st'
ONLINE_ST_FILES=${OUTPUT_PATH}/online_ST_files/
MODEL_PATH="${MODELS_PATH}/TransCoder_model_1.pth"

python -m codegen_sources.model.train \
## General parameters
--dump_path $DUMP_PATH \
--exp_name online_st \
--data_path $ONLINE_ST_FILES \
--unit_tests_path "${ONLINE_ST_FILES}/translated_tests.json" \

## Model parameter 
--encoder_only false \
--n_layers 0 \
--n_layers_decoder 6 \
--n_layers_encoder 6 \
--emb_dim 1024 \
--n_heads 8 \
--dropout '0.1' \
--lgs 'cpp_sa-java_sa-python_sa' \
--max_vocab 64000 \
--max_len 512 \
--reload_model '${MODEL_PATH},${MODEL_PATH}' \

## Optimization parameters
--optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.00003,weight_decay=0.01' \
--amp 2 \
--fp16 true \
--max_batch_size 128 \
--tokens_per_batch 4000 \
--epoch_size 2500 \
--max_epoch 10000000 \
--clip_grad_norm 1 \
--stopping_criterion 'valid_java_sa-python_sa_mt_comp_acc,25' \
--validation_metrics 'valid_java_sa-python_sa_mt_comp_acc,valid_java_sa-cpp_sa_mt_comp_acc,valid_python_sa-java_sa_mt_comp_acc,valid_python_sa-cpp_sa_mt_comp_acc,valid_cpp_sa-java_sa_mt_comp_acc,valid_cpp_sa-python_sa_mt_comp_acc' \
--has_sentence_ids 'valid|para,test|para,self_training|java_sa' \

## Evaluation parameters
--eval_bleu true \
--eval_computation true \
--generate_hypothesis true \
--eval_st false \
--eval_only false \

## self-training parameters
--st_steps 'java_sa-python_sa|cpp_sa' \
--st_beam_size 20 \
--lambda_st 1 \
--robin_cache false \
--st_sample_size 200 \
--st_limit_tokens_per_batch true \
--st_remove_proba '0.3' \
--st_sample_cache_ratio '0.5' \
--cache_init_path "${ONLINE_ST_FILES}/initial_cache/" \
--cache_warmup 500 