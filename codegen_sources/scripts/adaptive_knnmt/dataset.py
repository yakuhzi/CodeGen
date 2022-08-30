import os
import torch
import random
import numpy as np

from pathlib import Path
from torch.utils.data import random_split
from typing import List, Tuple
from tqdm import tqdm
from codegen_sources.model.translate import Translator

SEED=2022


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        parallel_functions: str, 
        cache_dir: str,
        translator: Translator, 
        language_pair: str, 
        phase: str, 
        samples: int
    ):
        self.parallel_functions = parallel_functions
        self.cache_dir = cache_dir
        self.translator = translator
        self.language_pair = language_pair
        self.phase = phase
        self.samples = samples

        self.src_language = language_pair.split("_")[0]
        self.tgt_language = language_pair.split("_")[1]

        self.features, self.scores, self.targets, self.inputs, self.outputs = self.make_dataset(parallel_functions)

    def __len__(self) -> int:
        return self.samples # len(self.features) - len(self.features) % self.batch_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features[index]
        scores = self.scores[index]
        target = self.targets[index]
        inputs = self.inputs[index]
        outputs = self.outputs[index]

        features = torch.from_numpy(features)
        scores = torch.from_numpy(scores)

        return features, scores, target, inputs, outputs

    def make_dataset(self, parallel_functions) -> Tuple[list, list, List[str]]:
        print(f"Building dataset for '{self.phase}'")
        configuration = f"{self.phase}_{SEED}_{self.samples}"
        cache_dir = os.path.join(self.cache_dir, self.language_pair, configuration)

        if os.path.exists(cache_dir):
            print(f"Using cached dataset for '{self.phase}'")
            features = np.load(os.path.join(cache_dir, "features.npy"))
            scores = np.load(os.path.join(cache_dir, "scores.npy"))
            targets = np.load(os.path.join(cache_dir, "targets.npy"))
            inputs = np.load(os.path.join(cache_dir, "inputs.npy"))
            outputs = np.load(os.path.join(cache_dir, "outputs.npy"))

            assert len(features) == len(scores) == len(targets) == len(inputs) == len(outputs) == self.samples
            return features, scores, targets, inputs, outputs

        dataset_size = len(parallel_functions)
        split_sizes = [int(dataset_size * 0.8), int(dataset_size * 0.1), int(dataset_size * 0.1)]

        if split_sizes[0] + split_sizes[1] + split_sizes[2] != len(parallel_functions):
            split_sizes[0] += 1

        train_set, val_set, test_set = random_split(
            parallel_functions, 
            split_sizes, 
            generator=torch.Generator().manual_seed(SEED)
        )

        if self.phase == "train":
            parallel_functions = train_set
        elif self.phase == "val":
            parallel_functions = val_set
        elif self.phase == "test":
            parallel_functions = test_set

        parallel_functions = random.Random(SEED).sample(list(parallel_functions), int(self.samples / 10))

        features = []
        scores = []
        targets = []
        inputs = []
        outputs = []

        with tqdm(total=len(parallel_functions)) as pbar:
            for src_sample, tgt_sample in parallel_functions:
                # tgt_samples = tgt_sample.split(" ")
                # tgt_sample = " ".join(tgt_samples[:random.Random(SEED).randrange(len(tgt_samples))])

                decoder_features, decoder_scores, decoder_targets, target_tokens, input_code, output_code = self.translator.get_features(
                    input_code=src_sample,
                    target_code=tgt_sample,
                    src_language=self.src_language,
                    tgt_language=self.tgt_language,
                    predict_single_token=False,
                    tokenized=True
                )

                for index, target in enumerate(decoder_targets[1:]):
                    features.append(decoder_features[index].cpu().detach().numpy())
                    scores.append(decoder_scores[index].cpu().detach().numpy())
                    targets.append(target.item())
                    inputs.append(input_code)
                    outputs.append(" ".join(output_code.split(" ")[1:index + 1]))

                pbar.update(1)

        features = np.array(random.Random(SEED).sample(features, self.samples))
        scores = np.array(random.Random(SEED).sample(scores, self.samples))
        targets = np.array(random.Random(SEED).sample(targets, self.samples))
        inputs = np.array(random.Random(SEED).sample(inputs, self.samples))
        outputs = np.array(random.Random(SEED).sample(outputs, self.samples))

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(cache_dir, "features.npy"), features)
        np.save(os.path.join(cache_dir, "scores.npy"), scores)
        np.save(os.path.join(cache_dir, "targets.npy"), targets)
        np.save(os.path.join(cache_dir, "inputs.npy"), inputs)
        np.save(os.path.join(cache_dir, "outputs.npy"), outputs)

        assert len(features) == len(scores) == len(targets) == len(inputs) == len(outputs) == self.samples
        return features, scores, targets, inputs, outputs
