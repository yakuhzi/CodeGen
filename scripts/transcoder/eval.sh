#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --time=8:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval
#SBATCH --output=eval_transcoder_%j.out

MODEL_PATH='models/transcolder/TransCoder_model_1.pth'
DUMP_PATH='dump/transcoder'
DATASET_PATH='dataset/transcoder/test'

python codegen_sources/model/train.py \
    --exp_name transcoder_eval \
    --dump_path "$DUMP_PATH" \
    --data_path "$DATASET_PATH" \
    --bt_steps 'python_sa-java_sa-python_sa,java_sa-python_sa-java_sa,python_sa-cpp_sa-python_sa,java_sa-cpp_sa-java_sa,cpp_sa-python_sa-cpp_sa,cpp_sa-java_sa-cpp_sa'    \
    --encoder_only False \
    --n_layers 0  \
    --n_layers_encoder 6  \
    --n_layers_decoder 6 \
    --emb_dim 1024  \
    --n_heads 8  \
    --lgs 'cpp_sa-java_sa-python_sa'  \
    --max_vocab 64000 \
    --gelu_activation false \
    --roberta_mode false  \
    --amp 2  \
    --fp16 true  \
    --tokens_per_batch 3000  \
    --max_batch_size 128 \
    --eval_bleu true \
    --eval_computation true \
    --has_sentence_ids "valid|para,test|para" \
    --generate_hypothesis true \
    --save_periodic 1 \
    --reload_model "$MODEL_PATH,$MODEL_PATH" \
    --reload_encoder_for_decoder false \
    --eval_only true \
    --n_sentences_eval 1500