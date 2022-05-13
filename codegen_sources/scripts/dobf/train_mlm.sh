python train.py 

## main parameters
--exp_name mlm \
--dump_path '<YOUR_DUMP_PATH>' \ 

## data / objectives
--data_path '<DATA_PATH>' \ 
--split_data_accross_gpu local \
--mlm_steps 'java,python' \
--add_eof_to_stream true \
--word_mask_keep_rand '0.8,0.1,0.1' \
--word_pred '0.15' \


## model
--encoder_only true \
--n_layers 12  \
--emb_dim 768  \
--n_heads 12  \
--lgs 'java-python' \
--max_vocab 64000 \
--gelu_activation true\
--roberta_mode true 

#optimization
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