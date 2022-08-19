from tqdm import tqdm

from ...model.translate import Translator
from .knnmt import KNNMT
from .load_functions import load_parallel_functions, load_validation_functions
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

        translator_path = f"models/transcoder_st/Online_ST_{src_language.title()}_{tgt_language.title()}.pth"
        translator_path = translator_path.replace("Cpp", "CPP")
        translator = Translator(translator_path, "data/bpe/cpp-java-python/codes", global_model=True)

        # Obtain features and targets from decoder
        for src_sample, tgt_sample in self.chunk:
            decoder_features, targets, target_tokens = translator.get_features(
                input_code=src_sample, 
                target_code=tgt_sample, 
                src_language=src_language, 
                tgt_language=tgt_language,
                tokenized=not self.is_validation
            )

            self.knnmt.add_to_datastore(decoder_features, targets, self.language_pair)
            self.pbar.update(1)


    def join(self):
        self.ctx.detach()
        threading.Thread.join(self)


def add_to_datastore(knnmt: KNNMT, parallel_functions, is_validation: bool=False):
    driver.init()

    for language_pair in LANGUAGE_PAIRS:
        print("#" * 10 + f" Creating Datastore for '{language_pair}' " + "#" * 10)

        functions = parallel_functions[language_pair]
        chunks = np.array_split(functions, driver.Device.count())

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


def train_datastore(knnmt: KNNMT):
    for language_pair in LANGUAGE_PAIRS:
        knnmt.train_datastore(language_pair)


def created_mixed_datastore(knnmt: KNNMT):
    for language_pair in LANGUAGE_PAIRS:
        datastore_keys = np.load(f"dump/knnmt/datastore/keys_{language_pair}.npy")
        datastore_keys_valid = np.load(f"dump/knnmt_valid/datastore/keys_{language_pair}.npy")

        datastore_values= np.load(f"dump/knnmt/datastore/values_{language_pair}.npy")
        datastore_values_valid= np.load(f"dump/knnmt_valid/datastore/values_{language_pair}.npy")

        datastore_keys = np.concatenate((datastore_keys, datastore_keys_valid), axis=0)
        datastore_values = np.concatenate((datastore_values, datastore_values_valid), axis=0)

        print("Keys", datastore_keys.shape)
        print("Values", datastore_values.shape)

        np.save(f"dump/knnmt_mixed/datastore/keys_{language_pair}.npy", datastore_keys)
        np.save(f"dump/knnmt_mixed/datastore/values_{language_pair}.npy", datastore_values)


knnmt = KNNMT()

# parallel_functions = load_parallel_functions()
# add_to_datastore(knnmt, parallel_functions)

# validation_functions = load_validation_functions()
# add_to_datastore(knnmt, validation_functions, is_validation=True)

# created_mixed_datastore(knnmt)
train_datastore(knnmt)
