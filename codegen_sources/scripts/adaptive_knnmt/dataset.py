import os
import torch
import random
import numpy as np

from pathlib import Path
from hashlib import sha256
from torch.utils.data import random_split
from typing import List, Tuple
from tqdm import tqdm
from codegen_sources.model.translate import Translator
from codegen_sources.scripts.knnmt.knnmt import KNNMT

SEED=2022


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        batch_size: int,
        parallel_functions: str, 
        cache_dir: str,
        knnmt: KNNMT, 
        translator: Translator, 
        language_pair: str, 
        phase: str, 
        samples: int
    ):
        self.batch_size = batch_size
        self.parallel_functions = parallel_functions
        self.cache_dir = cache_dir
        self.language_pair = language_pair
        self.phase = phase
        self.samples = samples

        self.knnmt = knnmt
        self.translator = translator

        self.src_language = language_pair.split("_")[0]
        self.tgt_language = language_pair.split("_")[1]

        self.features, self.scores, self.targets = self.make_dataset(parallel_functions, samples)

    def __len__(self) -> int:
        return self.samples # len(self.features) - len(self.features) % self.batch_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features[index]
        scores = self.scores[index]
        target = self.targets[index]

        features = torch.from_numpy(features)
        scores = torch.from_numpy(scores)

        return features, scores, target

    def make_dataset(self, parallel_functions, samples: int) -> Tuple[List[str], List[str]]:
        print(f"Building dataset for '{self.phase}'")
        configuration = f"{self.phase}_{SEED}_{samples}"
        cache_dir = os.path.join(self.cache_dir, self.language_pair, configuration)

        if os.path.exists(cache_dir):
            print(f"Using cached dataset for '{self.phase}'")
            features = np.load(os.path.join(cache_dir, "features.npy"))
            scores = np.load(os.path.join(cache_dir, "scores.npy"))
            targets = np.load(os.path.join(cache_dir, "targets.npy"))

            assert len(features) == len(scores) == len(targets) == samples
            return features, scores, targets

        dataset_size = len(parallel_functions)
        split_sizes = [int(dataset_size / 3), int(dataset_size / 3), int(dataset_size / 3)]

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

        parallel_functions = random.Random(SEED).sample(list(parallel_functions), int(samples / 10))

        features = []
        scores = []
        targets = []

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

                pbar.update(1)

        features = np.array(random.Random(SEED).sample(features, samples))
        scores = np.array(random.Random(SEED).sample(scores, samples))
        targets = np.array(random.Random(SEED).sample(targets, samples))

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(cache_dir, "features.npy"), features)
        np.save(os.path.join(cache_dir, "scores.npy"), scores)
        np.save(os.path.join(cache_dir, "targets.npy"), targets)

        assert len(features) == len(scores) == len(targets) == samples
        return features, scores, targets
