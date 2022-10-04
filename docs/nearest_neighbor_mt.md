# Nearest Neighbor Machine Translation

The approach of the nearest neighbor machine translation (kNN-MT) is based on the work of [urvashik/knnmt](https://github.com/urvashik/knnmt) and [zhengxxn/adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt).

## Create the Datastore
To be able to create the kNN-MT datastore, one needs a parallel corpus. To obtain the parallel corpus from TransCoder-ST follow the steps described [here](./parallel_corpus.md).

Run the following script to create the kNN-MT datastore for the parallel corpus, for the TransCoder-ST validation set and for a mixed datastore based on both sources:

```sh
codegen_sources/scripts/knnmt/create_datastore.sh
```

Note this can take a very long time. Consider splitting it up in multiple chunks or use a lot of GPUs for processing.

## Train Adaptive kNN-MT Model
To train the adaptive kNN-MT model for a language pair, run the following script:

```sh
codegen_sources/scripts/adaptive_knnmt/train/train_{src_language}_{tgt_language}.sh
```

The default arguments and hyperparameters can be seen in `codegen_sources/scripts/adaptive_knnmt/arguments.py`.

## Experiments
To run an experiment for a specific language pair, replace the respective placeholders and run the command.

### Vanilla Nearest Neighbor Translation
For the vanilla kNN-MT experiments, following arguments can be provided by modifying the `.sh` scripts:

1. `knnmt_dir`: Path to the directory containing the kNN-MT files (datastore and Faiss index)
2. `knnmt_temperature`: Temperature applied to the softmax over the kNN-MT predictions
3. `knnmt_tc_temperature`: Temperature applied to the softmax over the TC predictions when using kNN-MT
4. `knnmt_lambda`: Interpolation hyperparameter for weighting the kNN-MT and TC predictions
5. `knnmt_k`: Number of neighbors to retrieve from the kNN-MT datastore
6. `knnmt_restricted`: If the kNN-MT datastore should only be queried for functions that do not compile in the first place


#### Results With Datastore Created From Parallel Corpus

```sh
codegen_sources/scripts/knnmt/eval_parallel_corpus/eval_{src_language}_{tgt_language}.sh
```

#### Results With Datastore Created From TransCoder Validation Set

```sh
codegen_sources/scripts/knnmt/eval_validation_set/eval_{src_language}_{tgt_language}.sh
```

#### Results With Datastore Created From Both the Parallel Corpus and the TransCoder Validation Set

```sh
codegen_sources/scripts/knnmt/eval_mixed/eval_{src_language}_{tgt_language}.sh
```


### Adaptive Nearest Neighbor Translation
For the adaptive kNN-MT experiments, following arguments can be provided by modifying the `.sh` scripts:

1. `knnmt_dir`: Path to the directory containing the kNN-MT files (datastore and Faiss index)
2. `meta_k_checkpoint`: Path to the checkpoint of the trained Meta-k network

#### Results With Datastore Created From Parallel Corpus

```sh
codegen_sources/scripts/adaptive_knnmt/eval_parallel_corpus/eval_{src_language}_{tgt_language}.sh
```

#### Results With Datastore Created From TransCoder Validation Set

```sh
codegen_sources/scripts/adaptive_knnmt/eval_validation_set/eval_{src_language}_{tgt_language}.sh
```

#### Results With Datastore Created From Both the Parallel Corpus and the TransCoder Validation Set

```sh
codegen_sources/scripts/adaptive_knnmt/eval_mixed/eval_{src_language}_{tgt_language}.sh
```
