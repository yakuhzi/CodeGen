# TransCoder-ST Baseline
This page documents how the results for the baseline experiments can be obtained.  
For the baseline, a beam of 10 is used, picking the function with the highest probability in the end.

## Experiments
Run evaluation for a language pair:

```sh
codegen_sources/scripts/transcoder_st/eval/eval_{src_language}_{tgt_language}.sh
```

## Results
Beam search is used to generate 10 hypotheses, selecting the translation with the highest probability in the end.

|      Task      |   CA  | Evaluated | Success | Errors |
|----------------|:-----:|:---------:|:-------:|:------:|
|   C++ to Java  | 67,57 |    481    |   325   |   156  |
|  C++ to Python | 61,12 |    463    |   283   |   180  |
|   Java to C++  | 84,33 |    466    |   393   |   73   |
| Java to Python | 68,90 |    463    |   319   |   144  |
|  Python to C++ | 54,51 |    466    |   254   |   212  |
| Python to Java | 58,21 |    481    |   280   |   201  |
|     Overall    | 65,77 |    2820   |   1854  |   966  |