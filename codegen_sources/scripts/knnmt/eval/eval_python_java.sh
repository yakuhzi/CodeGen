#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=ev_pj_knn
#SBATCH --output=transcoder_st_python_java_knn_mixed_%j.log

MODEL_PATH='/pfs/work7/workspace/scratch/hd_tf268-code-gen/models/Online_ST_Python_Java.pth'
DUMP_PATH='/pfs/work7/workspace/scratch/hd_tf268-code-gen/dump/knnmt/eval/python_java'
DATASET_PATH='/pfs/work7/workspace/scratch/hd_tf268-code-gen/dataset/transcoder/test'

python -m codegen_sources.model.train \
    --dump_path "$DUMP_PATH" \
    --exp_name online_st \
    --data_path "$DATASET_PATH" \
    --encoder_only false \
    --n_layers 0 \
    --n_layers_decoder 6 \
    --n_layers_encoder 6 \
    --emb_dim 1024 \
    --n_heads 8 \
    --dropout '0.1' \
    --lgs 'python_sa-java_sa' \
    --max_vocab 64000 \
    --max_len 512 \
    --reload_model "${MODEL_PATH},${MODEL_PATH}" \
    --optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.00003,weight_decay=0.01' \
    --amp 2 \
    --fp16 true \
    --max_batch_size 128 \
    --tokens_per_batch 4000 \
    --epoch_size 2500 \
    --max_epoch 10000000 \
    --clip_grad_norm 1 \
    --stopping_criterion 'valid_python_sa-java_sa_mt_comp_acc,25' \
    --validation_metrics 'valid_python_sa-java_sa_mt_comp_acc' \
    --has_sentence_ids 'valid|para,test|para' \
    --eval_bleu true \
    --eval_computation true \
    --generate_hypothesis true \
    --eval_st false \
    --eval_only true \
    --st_steps 'python_sa-java_sa' \
    --st_beam_size 20 \
    --lambda_st 1 \
    --robin_cache false \
    --st_sample_size 200 \
    --st_limit_tokens_per_batch true \
    --st_remove_proba '0.3' \
    --st_sample_cache_ratio '0.5' \
    --knnmt_dir '/pfs/work7/workspace/scratch/hd_tf268-code-gen/knnmt' \
    --beam_size 1