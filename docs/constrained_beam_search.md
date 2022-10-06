# Constrained Beam Search
This page documents how the results for the experiments with constrained beam search can be obtained.

## Experiments
Run evaluation for a language pair:

```sh
codegen_sources/scripts/constrained/eval_{src_language}_{tgt_language}.sh
```

## Results
Beam search is used to generate 10 hypotheses. For the baseline, the translation with the highest probability is selected in the end. For the constrained ex- periments, the first syntactically correct function is selected.

| Task           | Original CA | Improved CA | Improvement | Original Errors | Original Compile-Time Errors | Fixed |
|----------------|:-----------:|:-----------:|:-----------:|:---------------:|:----------------------------:|:-----:|
|   C++ to Java  |    67,57    |    71,31    |     3,74    |       156       |              106             |   18  |
|  C++ to Python |    61,12    |    62,42    |     1,30    |       180       |              14              |   6   |
|   Java to C++  |    84,33    |    86,48    |     2,15    |        73       |              25              |   10  |
| Java to Python |    68,90    |    69,11    |     0,22    |       144       |               1              |   1   |
|  Python to C++ |    54,51    |    57,08    |     2,58    |       212       |              84              |   12  |
| Python to Java |    58,21    |    61,12    |     2,91    |       201       |              75              |   14  |
|     Overall    |    65,77    |    67,92    |     2,15    |       966       |              305             |   61  |
