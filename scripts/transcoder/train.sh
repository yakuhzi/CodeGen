#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --time=8:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval
#SBATCH --output=train_transcoder_%j.out

MODEL='TransCoder_model_1.pth'
MLM_MODEL='models/mlm/'
DUMP_PATH='dump/transcoder'
DATASET='dataset/transcoder'


python codegen_sources/model/train.py   

## main parameters
--exp_name transcoder \
--dump_path $DUMP_PATH \ 

## data / objectives
--data_path $DATASET \
--split_data_accross_gpu local \
--bt_steps 'python_sa-java_sa-python_sa,java_sa-python_sa-java_sa,python_sa-cpp_sa-python_sa,java_sa-cpp_sa-java_sa,cpp_sa-python_sa-cpp_sa,cpp_sa-java_sa-cpp_sa' \
--ae_steps 'python_sa,java_sa'  \
--lambda_ae '0:1,30000:0.1,100000:0'  \ 
--word_shuffle 3  \
--word_dropout '0.1' \ 
--word_blank '0.3'  \

## model  
--encoder_only False \
--n_layers 0  \
--n_layers_encoder 6  \
--n_layers_decoder 6 \
--emb_dim 1024  \
--n_heads 8  \
--lgs 'cpp_sa-java_sa-python_sa'  \
--max_vocab 64000 \
--gelu_activation false \
--roberta_mode false   \ 

## model reloading
--reload_model '${MLM_MODEL},${MLM_MODEL}'  \
--reload_encoder_for_decoder true \
--lgs_mapping 'cpp_sa:cpp,java_sa:java,python_sa:python'  \

## optimization
--amp 2  \
--fp16 true  \
--tokens_per_batch 3000  \
--group_by_size true \
--max_batch_size 128 \
--epoch_size 50000  \
--max_epoch 10000000  \
--split_data_accross_gpu global \
--optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01' \
--eval_bleu true \
--eval_computation true \
--has_sentence_ids "valid|para,test|para" \
--generate_hypothesis true \
--save_periodic 1 \
--validation_metrics 'valid_python_sa-java_sa_mt_comp_acc' 