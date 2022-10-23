from typing import Optional
from tqdm import tqdm

from ...model.translate import Translator
from .knnmt import KNNMT
from . import load_functions
import numpy as np
import threading
from pycuda import driver

LANGUAGE_PAIRS = [
    "cpp_java",
    "cpp_python",
    "java_cpp",
    "java_python",
    "python_cpp",
    "python_java",
]


class GPUThread(threading.Thread):
    def __init__(self, gpuid, knnmt, language_pair, chunk, pbar, is_validation):
        threading.Thread.__init__(self)

        self.ctx = driver.Device(gpuid).make_context()
        self.device = self.ctx.get_device()

        self.knnmt = knnmt
        self.language_pair = language_pair
        self.chunk = chunk
        self.pbar = pbar
        self.is_validation = is_validation

    def run(self):
        print("Add to dataset", self.getName(), self.device.name(), self.ctx.get_api_version())

        src_language = self.language_pair.split("_")[0]
        tgt_language = self.language_pair.split("_")[1]

        translator_path = f"models/Online_ST_{src_language.title()}_{tgt_language.title()}.pth"
        translator_path = translator_path.replace("Cpp", "CPP")
        translator = Translator(translator_path, "data/bpe/cpp-java-python/codes", global_model=True)

        # Obtain features and targets from decoder
        for src_sample, tgt_sample in self.chunk:
            decoder_features, _, targets, target_tokens, input_code, output_code = translator.get_features(
                input_code=src_sample,
                target_code=tgt_sample,
                src_language=src_language,
                tgt_language=tgt_language,
                tokenized=not self.is_validation
            )

            self.knnmt.add_to_datastore(decoder_features, targets, input_code, output_code, self.language_pair)
            self.pbar.update(1)

    def join(self):
        self.ctx.detach()
        threading.Thread.join(self)


def add_to_datastore(knnmt: KNNMT, parallel_functions, is_validation: bool=False):
    driver.init()

    for language_pair in parallel_functions.keys():
        print("#" * 10 + f" Creating Datastore for '{language_pair}' " + "#" * 10)

        # Get parallel functions for language pair and split into chunks
        functions = parallel_functions[language_pair]
        chunks = np.array_split(functions, driver.Device.count())

        # Add functions to kNN-MT datastore
        with tqdm(total=len(functions)) as pbar:
            threads = [
                GPUThread(index, knnmt, language_pair, chunk, pbar, is_validation)
                for index, chunk in enumerate(chunks)
            ]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

        knnmt.save_datastore(language_pair)


def train_datastore(knnmt: KNNMT, language_pair: Optional[str]):
    if language_pair is not None:
        knnmt.train_datastore(language_pair)
        return

    for language_pair in LANGUAGE_PAIRS:
        knnmt.train_datastore(language_pair)


def created_mixed_datastore(knnmt_dir: str):
    for language_pair in LANGUAGE_PAIRS:
        # Load datastore keys from the parallel corpus and from the validation set
        datastore_keys = np.load(f"{knnmt_dir}/parallel_corpus/datastore/keys_{language_pair}.npy")
        datastore_keys_valid = np.load(f"{knnmt_dir}/validation_set/datastore/keys_{language_pair}.npy")

        # Load datastore values from the parallel corpus and from the validation set
        datastore_values = np.load(f"{knnmt_dir}/knnmt_parallel_corpus/datastore/values_{language_pair}.npy")
        datastore_values_valid = np.load(f"{knnmt_dir}/validation_set/datastore/values_{language_pair}.npy")

        # Load datastore inputs from the parallel corpus and from the validation set
        datastore_inputs = np.load(f"{knnmt_dir}/parallel_corpus/datastore/inputs_{language_pair}.npy")
        datastore_inputs_valid = np.load(f"{knnmt_dir}/validation_set/datastore/inputs_{language_pair}.npy")

        # Concatenate datastores
        datastore_keys = np.concatenate((datastore_keys, datastore_keys_valid), axis=0)
        datastore_values = np.concatenate((datastore_values, datastore_values_valid), axis=0)
        datastore_inputs = np.concatenate((datastore_inputs, datastore_inputs_valid), axis=0)

        print("Keys", datastore_keys.shape)
        print("Values", datastore_values.shape)
        print("Inputs", datastore_inputs.shape)

        # Save datastores
        np.save(f"{knnmt_dir}/mixed/datastore/keys_{language_pair}.npy", datastore_keys)
        np.save(f"{knnmt_dir}/mixed/datastore/values_{language_pair}.npy", datastore_values)
        np.save(f"{knnmt_dir}/mixed/datastore/inputs_{language_pair}.npy", datastore_inputs)


# Create kNN-MT datastore from parallel corpus
knnmt_parallel_corpus = KNNMT("out/knnmt/parallel_corpus")
parallel_functions = load_functions.load_parallel_functions("data/parallel_corpus")
add_to_datastore(knnmt_parallel_corpus, parallel_functions)
train_datastore(knnmt_parallel_corpus)

# Create kNN-MT datastore from TransCoder validation set
knnmt_valid = KNNMT("out/knnmt/validation_set")
validation_functions = load_functions.load_validation_functions("data/test_dataset")
add_to_datastore(knnmt_valid, validation_functions, is_validation=True)
train_datastore(knnmt_valid)

# Create kNN-MT datastore from TransCoder first half of validation set for Meta-k network training
knnmt_valid_half = KNNMT("out/knnmt/validation_set_half")
validation_functions_half = load_functions.load_validation_functions("data/test_dataset", half=1)
add_to_datastore(knnmt_valid_half, validation_functions_half, is_validation=True)
train_datastore(knnmt_valid_half)

# Create kNN-MT datastore from parallel corpus and TransCoder validation set
created_mixed_datastore("out/knnmt")
knnmt_mixed = KNNMT("out/knnmt/mixed")
train_datastore(knnmt_mixed)
