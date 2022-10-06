# Rule-Based Error Corrections
This page documents how the results for the experiments with the rule-based corrections can be obtained.

## Experiments
Run evaluation for a language pair:

```sh
codegen_sources/scripts/corrections/eval/eval_{src_language}_{tgt_language}.sh
```

If you only want to see the results for initially failing functions:
```sh
python -m codegen_sources.scripts.corrections.corrections -s {src_language} -t {tgt_language}
```

Note: For this to work you need to run the evaluation of the original TransCoder-ST model at least once for the specified language pair.

## Results
Beam search is used to generate 10 hypotheses, selecting the translation with the highest probability in the end.

| Task           | Original CA | Improved CA | Improvement | Original Errors | Fixed | Corrupted |
|----------------|:-----------:|:-----------:|:-----------:|:---------------:|:-----:|:---------:|
|   C++ to Java  |    67,57    |    75,68    |     8,11    |       156       |   40  |     1     |
|  C++ to Python |    61,12    |    64,36    |     3,24    |       180       |   16  |     1     |
|   Java to C++  |    84,33    |    86,70    |     2,36    |        73       |   12  |     1     |
| Java to Python |    68,90    |    71,71    |     2,81    |       144       |   15  |     2     |
|  Python to C++ |    54,51    |    58,37    |     3,86    |       212       |   18  |     0     |
| Python to Java |    58,21    |    60,08    |     1,87    |       201       |   10  |     1     |
|     Overall    |    65,77    |    69,48    |     3,71    |       966       |  111  |     6     |
