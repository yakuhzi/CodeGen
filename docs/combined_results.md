# Combined Results
This page documents how the results for the experiments combining multiple techniques can be obtained.

## Experiments
To run an experiment for a specific language pair, replace the respective placeholders and run the command.


### All Techniques Combined
```sh
codegen_sources/scripts/combined/all/eval_{src_language}_{tgt_language}.sh
```

### Constrained Beam Search + Adaptive kNN-MT
```sh
codegen_sources/scripts/combined/constrained_knnmt/eval_{src_language}_{tgt_language}.sh
```

### Rule-Based Corrections + Constrained Beam Search
```sh
codegen_sources/scripts/combined/corrections_constrained/eval_{src_language}_{tgt_language}.sh
```

### Rule-Based Corrections + Adaptive kNN-MT
```sh
codegen_sources/scripts/combined/corrections_knnmt/eval_{src_language}_{tgt_language}.sh
```

## Results
Beam search is used to generate 10 hypotheses. For the experiments including constrained beam search, the first syntactically correct function is selected. For the other experiments, the translation with the highest probability is selected in the end.

| Task           | Original CA | Corrections + Constrained | Fixed | Corrections + A-kNN-MT | Fixed | Constrained + A-kNN-MT | Fixed | Corrections + Constrained + A-kNN-MT | Fixed |
|----------------|:-----------:|:-------------------------:|:-----:|:----------------------:|:-----:|:----------------------:|:-----:|:------------------------------------:|:-----:|
|   C++ to Java  |    67,57    |           77,96           |   50  |          77,55         |   48  |          75,26         |   37  |                 79,42                |   57  |
|  C++ to Python |    61,12    |           65,44           |   20  |          65,23         |   19  |          62,42         |   6   |                 65,44                |   20  |
|   Java to C++  |    84,33    |           88,20           |   18  |          87,55         |   15  |          86,48         |   10  |                 88,20                |   18  |
| Java to Python |    68,90    |           71,92           |   14  |          71,92         |   14  |          69,11         |   1   |                 71,92                |   14  |
|  Python to C++ |    54,51    |           58,80           |   20  |          58,37         |   18  |          58,37         |   18  |                 59,66                |   24  |
| Python to Java |    58,21    |           62,37           |   20  |          59,88         |   8   |          61,33         |   15  |                 62,16                |   19  |
|     Overall    |    65,77    |           70,78           |  142  |          70,08         |  122  |          68,83         |   87  |                 71,13                |  152  |