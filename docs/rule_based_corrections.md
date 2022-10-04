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
