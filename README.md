# Analysis and Optimization of Unsupervised Code-to-Code Translation

This repository contains the code and resources of the master's thesis `Analysis and Optimization of Unsupervised Code-to-Code Translation` at the university of Heidelberg 2022. It is a fork of the original repository [CodeGen](https://github.com/facebookresearch/CodeGen) from Facebook, which provided most of the code and the pretrained models for [TransCoder](https://arxiv.org/pdf/2006.03511.pdf), [DOBF](https://arxiv.org/pdf/2102.07492.pdf) and [TransCoder-ST](https://arxiv.org/pdf/2110.06773.pdf).

Almost all code and scripts that were added during the master thesis can be found under [codegen_sources/scripts](codegen_sources/scripts).

## Setup

### Repository
Run the following command to clone the repository

```sh
git clone https://github.com/yakuhzi/c2c-translation.git
cd c2c-translation
```

### Install dependencies
Run the following script to install all required dependencies.

```sh
install_env.sh
```

The script will also download the pretrained TransCoder and TransCoder-ST models and the validation and test set for evaluation.

## Experiments & Results

- [TransCoder-ST Baseline](docs/baseline.md)
- [Rule-Based Error Corrections](docs/rule_based_corrections.md)
- [Constrained Beam Search](docs/constrained_beam_search.md)
- [Nearest Neighbor Machine Translation](docs/nearest_neighbor_mt.md)
- [Combined Results](docs/combined_results.md)

## Other Work

- [Attention Weight Analysis](docs/attention_weights.md)
- [TCRules](docs/tc_rules.md)
- [Snippets](docs/snippets.md)

## License
The validation and test parallel datasets from GeeksForGeeks, and the evaluation scripts under [data/transcoder_evaluation_gfg](data/transcoder_evaluation_gfg) are released under the Creative Commons Attribution-ShareAlike 2.0 license. See https://creativecommons.org/licenses/by-sa/2.0/ for more information.

The rest of the repository is under the MIT license. See [LICENSE](LICENSE) for more details.
