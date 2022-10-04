# TransCoder-ST Baseline
This page documents how the results for the baseline experiments can be obtained.  
For the baseline, a beam of 10 is used, picking the function with the highest probability in the end.

## Experiments
Run evaluation for a language pair:

```sh
codegen_sources/scripts/transcoder_st/eval/eval_{src_language}_{tgt_language}.sh
```
