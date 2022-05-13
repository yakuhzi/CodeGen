python -m codegen_sources.preprocessing.preprocess 
<DATASET_PATH>                                                          # folder containing raw data i.e json.gz
--langs java python                                                     # languages to prepocess
--mode obfuscation                                                      # dataset mode
--local True                                                            # Run on your local machine if True. If False run on a cluster (requires submitit setup)
--bpe_mode fast_bpe
--fastbpe_code_path <BPE_PATH>                                          # This can either be the bpe codes we provide in data/bpe/cpp-java-python/codes or codes learnt from monolingual dataset mode
--train_splits NGPU                                                     # nb of splits for training data - corresponds to the number of GPU you have