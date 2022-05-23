#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --time=1:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_mlm
#SBATCH --output=transcoder_train_mlm_%j.out

DUMP_PATH='dump/transcoder/train_mlm'
DATASET_PATH=$(ws_find code-gen)/dataset/test/XLM-syml

python -m codegen_sources.model.train \
--exp_name mlm \
--dump_path "$DUMP_PATH" \
--data_path "$DATASET_PATH" \
--split_data_accross_gpu local \
--mlm_steps 'cpp,java,python' \
--add_eof_to_stream true \
--word_mask_keep_rand '0.8,0.1,0.1' \
--word_pred '0.15' \
--encoder_only true \
--n_layers 6  \
--emb_dim 1024  \
--n_heads 8  \
--lgs 'cpp-java-python' \
--max_vocab 64000 \
--gelu_activation false \
--roberta_mode false \
--amp 2  \
--fp16 true  \
--batch_size 32 \
--bptt 512 \
--epoch_size 100000 \
--max_epoch 100000 \
--split_data_accross_gpu global \
--optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01' \
--save_periodic 0 \
--validation_metrics _valid_mlm_ppl \
--stopping_criterion '_valid_mlm_ppl,10' 