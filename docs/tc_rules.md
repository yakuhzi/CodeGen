# TCRules

## Idea
Given a set of rules that are able to translate lines of code between different programming languages on a high level, one can make use of them in order to fix corrupted lines of code in generated translations of the TransCoder-ST model. The assumption here is that a line of code in the source sequence always matches with a line of code in the target sequence and there exists some generic rules that are able to translate either line into the other. If the corrupted line is identified, one can try to find the matching line in the source sequence and apply the generic rule to obtain the correct translation.

## Experiments
Run the following command to see an example where a corrupt translation was fixed using the described approach.

```sh
python -m codegen_sources.scripts.tc_rules.correct
```