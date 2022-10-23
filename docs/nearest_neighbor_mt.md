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

The arguments and hyperparameters are specified [here](../codegen_sources/scripts/adaptive_knnmt/arguments.py):

- `learning_rate`: Initial learning rate
- `adam_betas`: Beta 1 and 2 of ADAM optimizer
- `hidden_size`: Hidden size of FFN layer
- `batch_size`: Batch size of the data loader
- `max_k`: Maximum number of neighbors to retrieve from the datastore
- `tc_k`: Number of scores to consider from the TransCoder models
- `knn_temperature`: Softmax temperature applied on kNN-MT distances
- `tc_temperature`: Softmax temperature applied on TransCoder scores

## One-Click Demo
A one click demo for the generation of translations using the kNN-MT datastores only is available in a jupyter notebook [here](../codegen_sources/scripts/knnmt/one_click_demo.ipynb).

Make sure you used the created conda environment `c2c-translation`. To add it as a kernel for your notebook, run the following commands:

```sh
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=c2c-translation
```

## Experiments
To run an experiment for a specific language pair, replace the respective placeholders and run the command.

### Vanilla Nearest Neighbor Translation
For the vanilla kNN-MT experiments, following arguments can be provided by modifying the `.sh` scripts:

- `knnmt_dir`: Path to the directory containing the kNN-MT files (datastore and Faiss index)
- `knnmt_temperature`: Temperature applied to the softmax over the kNN-MT predictions
- `knnmt_tc_temperature`: Temperature applied to the softmax over the TC predictions when using kNN-MT
- `knnmt_lambda`: Interpolation hyperparameter for weighting the kNN-MT and TC predictions
- `knnmt_k`: Number of neighbors to retrieve from the kNN-MT datastore
- `knnmt_restricted`: If the kNN-MT datastore should only be queried for functions that do not compile in the first place


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

- `knnmt_dir`: Path to the directory containing the kNN-MT files (datastore and Faiss index)
- `meta_k_checkpoint`: Path to the checkpoint of the trained Meta-k network
- `knnmt_restricted`: If the kNN-MT datastore should only be queried for functions that do not compile in the first place

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
Results of the vanilla kNN-MT approach by combining the predictions of the kNN-MT datastores with the predictions of TransCoder-ST for all functions  (`knnmt_restricted` is set to false).

Hyperparameters used for the vanilla kNN-MT experiments (defined [here](../codegen_sources/model/train.py)):

- `knnmt_temperature`: 10
- `knnmt_tc_temperature`: 5
- `knnmt_lambda`: 0.5
- `knnmt_k`: 8
- `knnmt_restricted`: false

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
Results of using only the kNN-MT datastore predictions for translation and completely ignoring TransCoder-ST (`knnmt_lambda` is set to 1 and `knnmt_restricted` is set to false).

Hyperparameters used for the experiments with only the kNN-MT datastores (defined [here](../codegen_sources/model/train.py)):

- `knnmt_temperature`: 10
- `knnmt_tc_temperature`: 5
- `knnmt_lambda`: 1
- `knnmt_k`: 8
- `knnmt_restricted`: false

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
Results of the vanilla kNN-MT approach, but restricting it in a way that the datastores are only queried in cases where the generated functions of TransCoder-ST do not compile in the first place (`knnmt_restricted` is set to true).

Hyperparameters used for the restricted vanilla kNN-MT experiments (defined [here](../codegen_sources/model/train.py)):

- `knnmt_temperature`: 10
- `knnmt_tc_temperature`: 5
- `knnmt_lambda`: 0.5
- `knnmt_k`: 8
- `knnmt_restricted`: true


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
Results of the adaptive kNN-MT approach, but restricting it in a way that the datastores are only queried in cases where the generated functions of TransCoder-ST do not compile in the first place (`knnmt_restricted` is set to true).

Hyperparameters set at evaluation time (defined [here](../codegen_sources/model/train.py)):

- `knnmt_restricted`: true

Hyperparameters set at training time (defined [here](../codegen_sources/scripts/adaptive_knnmt/arguments.py)):

- `learning_rate`: 1e-05
- `adam_betas`: 0.9, 0.98
- `hidden_size`: 32
- `batch_size`: 32
- `max_k`: 32
- `tc_k`: 32
- `knn_temperature`: 10
- `tc_temperature`: 3 (except for Java to C++ where it is set to 5)

Checkpoints used:
- `C++ to Java`: [BS32_KT10_TT3_MK32_TK32_HS32_LR1e-05_B0.9-0.98/602852/best-epoch=59.ckpt](../out/adaptive_knnmt/checkpoints/cpp_java/BS32_KT10_TT3_MK32_TK32_HS32_LR1e-05_B0.9-0.98/602852/best-epoch=59.ckpt)
- `C++ to Python`: [BS32_KT10_TT3_MK32_TK32_HS32_LR1e-05_B0.9-0.98/386692/best-epoch=96.ckpt](../out/adaptive_knnmt/checkpoints/cpp_python/BS32_KT10_TT3_MK32_TK32_HS32_LR1e-05_B0.9-0.98/386692/best-epoch=96.ckpt)
- `Java to C++`: [BS32_KT10_TT5_MK32_TK32_HS64_LR1e-05_B0.9-0.98/684418/best-epoch=87.ckpt](../out/adaptive_knnmt/checkpoints/java_cpp/BS32_KT10_TT5_MK32_TK32_HS64_LR1e-05_B0.9-0.98/684418/best-epoch=87.ckpt)
- `Java to Python`: [BS32_KT10_TT3_MK32_TK32_HS32_LR1e-05_B0.9-0.98/559204/best-epoch=81.ckpt](../out/adaptive_knnmt/checkpoints/java_python/BS32_KT10_TT3_MK32_TK32_HS32_LR1e-05_B0.9-0.98/559204/best-epoch=81.ckpt)
- `Python to C++`: [BS32_KT10_TT3_MK32_TK32_HS32_LR1e-05_B0.9-0.98/637986/best-epoch=85.ckpt](../out/adaptive_knnmt/checkpoints/python_cpp/BS32_KT10_TT3_MK32_TK32_HS32_LR1e-05_B0.9-0.98/637986/best-epoch=85.ckpt)
- `Python to Java`: [BS32_KT10_TT3_MK32_TK32_HS32_LR1e-05_B0.9-0.98/184136/best-epoch=70.ckpt](../out/adaptive_knnmt/checkpoints/python_java/BS32_KT10_TT3_MK32_TK32_HS32_LR1e-05_B0.9-0.98/184136/best-epoch=70.ckpt)

| Task           | Original CA | kNN-MT<sub>PC</sub> | Fixed | kNN-MT<sub>VAL</sub> | Fixed | kNN-MT<sub>PC+VAL</sub> | Fixed |
|----------------|:-----------:|:-------------------:|:-----:|:--------------------:|:-----:|:-----------------------:|:-----:|
|   C++ to Java  |    67,57    |        69,02        |   7   |         71,93        |   21  |          73,39          |   28  |
|  C++ to Python |    61,12    |        61,77        |   3   |         61,56        |   2   |          61,99          |   4   |
|   Java to C++  |    84,33    |        85,19        |   4   |         85,19        |   4   |          85,84          |   7   |
| Java to Python |    68,90    |        68,90        |   0   |         69,11        |   1   |          69,11          |   1   |
|  Python to C++ |    54,51    |        56,44        |   9   |         56,44        |   9   |          57,51          |   14  |
| Python to Java |    58,21    |        58,63        |   2   |         59,67        |   7   |          60,08          |   9   |
|     Overall    |    65,77    |        66,66        |   25  |         67,32        |   44  |          67,99          |   63  |