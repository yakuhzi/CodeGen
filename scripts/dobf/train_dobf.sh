python train.py 

## main parameters
--exp_name deobfuscation \
--dump_path '<YOUR_DUMP_PATH>' \

## data / objectives
--data_path '<DATA_PATH>' \
--split_data_accross_gpu local \
--do_steps 'python_obfuscated-python_dictionary,java_obfuscated-java_dictionary' \
--obf_proba '0.5' \
--ae_steps 'python_obfuscated,java_obfuscated' \
--mask_length poisson \
--word_shuffle 3  \
--word_dropout '0.1' \
--word_blank '0.3' \

## model
--encoder_only False \
--n_layers 0  \
--n_layers_encoder 12  \
--n_layers_decoder 6 \
--emb_dim 768  \
--n_heads 12  \
--lgs 'python_dictionary-python_obfuscated-java_dictionary-java_obfuscated' \
--max_vocab 64000 \
--gelu_activation true \
--roberta_mode true \ 
 
## model reloading
--reload_model '<PATH_TO_MLM_MODEL>,' \
--lgs_mapping 'python_dictionary:python,python_obfuscated:python,java_dictionary:java,java_obfuscated:java' \

## optimization
--amp 2  \
--fp16 true  \
--tokens_per_batch 3000  \
--group_by_size true \
--max_batch_size 128 \
--max_len 2000 \
--epoch_size 50000  \
--max_epoch 10000000  \
--split_data_accross_gpu global \
--optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01' \
--eval_bleu true \
--eval_subtoken_score true \
--save_periodic 10 \
--validation_metrics 'valid_obf_proba_#obf_proba_mt_subtoken_F1' \
--stopping_criterion 'valid_obf_proba_#obf_proba_mt_subtoken_F1,10'