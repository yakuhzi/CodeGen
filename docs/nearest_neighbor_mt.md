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

### Original Nearest Neighbor Translation
Run evaluation for a language pair:

```sh
codegen_sources/scripts/knnmt/eval/eval_{src_language}_{tgt_language}.sh
```

### Adaptive Nearest Neighbor Translation
Run evaluation for a language pair:

```sh
codegen_sources/scripts/adaptive_knnmt/eval/eval_{src_language}_{tgt_language}.sh
```