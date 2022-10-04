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
