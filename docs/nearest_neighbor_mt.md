# Nearest Neighbor Machine Translation (kNN-MT)

The approach of the nearest neighbor machine translation (kNN-MT) is based on the work of [urvashik/knnmt](https://github.com/urvashik/knnmt) and [zhengxxn/adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt).

## Create the kNN-MT Datastore
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

The default arguments and hyperparameters are specified [here](`../codegen_sources/scripts/adaptive_knnmt/arguments.py`).

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
3. `knnmt_restricted`: If the kNN-MT datastore should only be queried for functions that do not compile in the first place

The other parameters are already specified at training time (see [here](#train-adaptive-knn-mt-model)).
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


### Using the kNN-MT Datastore as a Self-Learning Error Correction Mechanism

Specify the target language pair and the path to the kNN-MT datastore in [here](../codegen_sources/scripts/knnmt/corrections.py).  
Then run the following script to add all the initially failing functions to the datastore (you might want to use a copy of the datastore):

```sh
python -m codegen_sources.scripts.knnmt.corrections
```

Finally, run the evaluation again using the just extended datastore, for example:

```sh
codegen_sources/scripts/knnmt/eval_mixed/eval_{src_language}_{tgt_language}.sh
```

Make sure you provided path to the extended kNN-MT datastore in the `.sh` file.

## Results
Beam search is used to generate 10 hypotheses, selecting the translation with the highest probability in the end.

### Vanilla kNN-MT Results
Results of the vanilla kNN-MT approach by combining the predictions of the kNN-MT datastores with the predictions of TransCoder-ST for all functions.

| Task           | Original CA | kNN-MT<sub>PC</sub> | kNN-MT<sub>VAL</sub> | kNN-MT<sub>PC+VAL</sub> |
|----------------|:-----------:|:-------------------:|:--------------------:|:-----------------------:|
|   C++ to Java  |    67,57    |        55,09        |         39,09        |          58,42          |
|  C++ to Python |    61,12    |        36,72        |         27,86        |          44,06          |
|   Java to C++  |    84,33    |        65,88        |         39,49        |          65,02          |
| Java to Python |    68,90    |        49,03        |         29,16        |          55,29          |
|  Python to C++ |    54,51    |        28,76        |         19,74        |          31,12          |
| Python to Java |    58,21    |        38,88        |         22,04        |          36,59          |
|     Overall    |    65,77    |        45,73        |         29,56        |          48,42          |

### kNN-MT Results Only
Results of using only the kNN-MT datastore predictions for translation and completely ignoring TransCoder-ST ($\lambda = 1$).

| Task           | Original CA | kNN-MT<sub>PC</sub> | kNN-MT<sub>VAL</sub> | kNN-MT<sub>PC+VAL</sub> |
|----------------|:-----------:|:-------------------:|:--------------------:|:-----------------------:|
|   C++ to Java  |    67,57    |        53,01        |         36,17        |          53,43          |
|  C++ to Python |    61,12    |        34,34        |         23,33        |          40,60          |
|   Java to C++  |    84,33    |        64,16        |         37,98        |          60,30          |
| Java to Python |    68,90    |        47,52        |         26,78        |          53,13          |
|  Python to C++ |    54,51    |        26,61        |         17,17        |          27,25          |
| Python to Java |    58,21    |        36,38        |         19,54        |          33,89          |
|     Overall    |    65,77    |        43,67        |         26,83        |          44,77          |

### Vanilla kNN-MT Results (Restricted)
Results of the vanilla kNN-MT approach, but restricting it in a way that the datastores are only queried in cases where the generated functions of TransCoder-ST do not compile in the first place.

| Task           | Original CA | kNN-MT<sub>PC</sub> | Fixed | kNN-MT<sub>VAL</sub> | Fixed | kNN-MT<sub>PC+VAL</sub> | Fixed |
|----------------|:-----------:|:-------------------:|:-----:|:--------------------:|:-----:|:-----------------------:|:-----:|
|   C++ to Java  |    67,57    |        68,82        |   6   |         69,65        |   10  |          72,14          |   22  |
|  C++ to Python |    61,12    |        61,56        |   2   |         61,77        |   3   |          61,56          |   2   |
|   Java to C++  |    84,33    |        85,19        |   4   |         84,98        |   3   |          85,62          |   6   |
| Java to Python |    68,90    |        68,90        |   0   |         68,90        |   0   |          69,11          |   1   |
|  Python to C++ |    54,51    |        55,58        |   5   |         55,36        |   4   |          57,08          |   12  |
| Python to Java |    58,21    |        58,42        |   1   |         59,04        |   4   |          59,46          |   6   |
|     Overall    |    65,77    |        66,41        |   18  |         66,62        |   24  |          67,50          |   49  |

### Adaptive kNN-MT Results (Restricted)
Results of the adaptive kNN-MT approach, but restricting it in a way that the datastores are only queried in cases where the generated functions of TransCoder-ST do not compile in the first place.

| Task           | Original CA | kNN-MT<sub>PC</sub> | Fixed | kNN-MT<sub>VAL</sub> | Fixed | kNN-MT<sub>PC+VAL</sub> | Fixed |
|----------------|:-----------:|:-------------------:|:-----:|:--------------------:|:-----:|:-----------------------:|:-----:|
|   C++ to Java  |    67,57    |        69,02        |   7   |         71,93        |   21  |          73,39          |   28  |
|  C++ to Python |    61,12    |        61,77        |   3   |         61,56        |   2   |          61,99          |   4   |
|   Java to C++  |    84,33    |        85,19        |   4   |         85,19        |   4   |          85,84          |   7   |
| Java to Python |    68,90    |        68,90        |   0   |         69,11        |   1   |          69,11          |   1   |
|  Python to C++ |    54,51    |        56,44        |   9   |         56,44        |   9   |          57,51          |   14  |
| Python to Java |    58,21    |        58,63        |   2   |         59,67        |   7   |          60,08          |   9   |
|     Overall    |    65,77    |        66,66        |   25  |         67,32        |   44  |          67,99          |   63  |