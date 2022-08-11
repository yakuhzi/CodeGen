from tqdm import tqdm
from ...model.translate import Translator
from .knnmt import KNNMT
from hashlib import sha256
import numpy as np
import threading
from pycuda import driver


DATASET_PATH = "dump/transcoder_st"
LANGUAGE_PAIRS = [
    "cpp_java",
    "cpp_python",
    "java_cpp",
    "java_python",
    "python_cpp",
    "python_java",
]


class GPUThread(threading.Thread):
    def __init__(self, gpuid, knnmt, language_pair, chunk, pbar):
        threading.Thread.__init__(self)

        self.ctx = driver.Device(gpuid).make_context()
        self.device = self.ctx.get_device()

        self.knnmt = knnmt
        self.language_pair = language_pair
        self.chunk = chunk
        self.pbar = pbar


    def run(self):
        print("run", self.getName(), self.device.name(), self.ctx.get_api_version())

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
                tokenized=True
            )

            self.knnmt.add_to_datastore(decoder_features, targets, self.language_pair)
            self.pbar.update(1)


    def join(self):
        self.ctx.detach()
        threading.Thread.join(self)


def load_parallel_functions():
    cpp_java_cpp_functions = []
    cpp_java_java_functions = []
    cpp_python_cpp_functions = []
    cpp_python_python_functions = []
    java_python_java_functions = []
    java_python_python_functions = []


    for i in range(145):
        cpp_java_cpp_file = open(f"{DATASET_PATH}/dataset_{i}/offline_dataset/train.cpp_sa-java_sa.cpp_sa.bpe", "r")
        cpp_java_cpp_functions += cpp_java_cpp_file.readlines()

        cpp_java_java_file = open(f"{DATASET_PATH}/dataset_{i}/offline_dataset/train.cpp_sa-java_sa.java_sa.bpe", "r")
        cpp_java_java_functions += cpp_java_java_file.readlines()

        cpp_python_cpp_file = open(f"{DATASET_PATH}/dataset_{i}/offline_dataset/train.cpp_sa-python_sa.cpp_sa.bpe", "r")
        cpp_python_cpp_functions += cpp_python_cpp_file.readlines()

        cpp_python_python_file = open(f"{DATASET_PATH}/dataset_{i}/offline_dataset/train.cpp_sa-python_sa.python_sa.bpe", "r")
        cpp_python_python_functions += cpp_python_python_file.readlines()

        java_python_java_file = open(f"{DATASET_PATH}/dataset_{i}/offline_dataset/train.java_sa-python_sa.java_sa.bpe", "r")
        java_python_java_functions += java_python_java_file.readlines()

        java_python_python_file = open(f"{DATASET_PATH}/dataset_{i}/offline_dataset/train.java_sa-python_sa.python_sa.bpe", "r")
        java_python_python_functions += java_python_python_file.readlines()

    parallel_functions = {}
    parallel_functions["cpp_java"] = list(zip(cpp_java_cpp_functions, cpp_java_java_functions))
    parallel_functions["cpp_python"] = list(zip(cpp_python_cpp_functions, cpp_python_python_functions))
    parallel_functions["java_cpp"] = list(zip(cpp_java_java_functions, cpp_java_cpp_functions))
    parallel_functions["java_python"] = list(zip(java_python_java_functions, java_python_python_functions))
    parallel_functions["python_cpp"] = list(zip(cpp_python_python_functions, cpp_python_cpp_functions))
    parallel_functions["python_java"] = list(zip(java_python_python_functions, java_python_java_functions))

    parallel_functions = deduped_parallel_functions(parallel_functions)
    return parallel_functions


def deduped_parallel_functions(parallel_functions):
    deduped_functions = {}

    for language_pair in LANGUAGE_PAIRS:
        function_pairs = parallel_functions[language_pair]
        deduped_pairs = []
        hashes = {}

        for pair in function_pairs:
            source, target = pair
            hash = sha256(source.encode("utf8") + target.encode("utf8")).hexdigest()

            if hashes.get(hash) is None:
                hashes[hash] = pair
                deduped_pairs.append(pair)

        deduped_functions[language_pair] = deduped_pairs

        print(f"Size of '{language_pair}' dataset: {len(function_pairs)}")
        print(f"Size of deduped '{language_pair}' dataset: {len(deduped_pairs)}")

    return deduped_functions


def create_datastore(knnmt: KNNMT):
    parallel_functions = load_parallel_functions()
    driver.init()

    for language_pair in LANGUAGE_PAIRS:
        print("#" * 10 + f" Creating Datastore for '{language_pair}' " + "#" * 10)

        functions = parallel_functions[language_pair]
        chunks = np.array_split(functions, driver.Device.count())

        with tqdm(total=len(functions)) as pbar:
            threads = [GPUThread(index, knnmt, language_pair, chunk, pbar) for index, chunk in enumerate(chunks)]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

        knnmt.save_datastore(language_pair)


def train_datastore(knnmt: KNNMT):
    for language_pair in LANGUAGE_PAIRS:
        knnmt.train_datastore(language_pair)


def output_sample(knnmt: KNNMT):
    src_language = "cpp"
    tgt_language = "java"

    translator_path = f"models/transcoder_st/Online_ST_{src_language.title()}_{tgt_language.title()}.pth"
    translator = Translator(
        translator_path.replace("Cpp", "CPP"), "data/bpe/cpp-java-python/codes", global_model=True
    )

    source = translator.tokenize("static boolean isEven(int n) { return (!(n & 1)); }", src_language)
    # generated = translator.tokenize("static int summingSeries(long n) { return (int) Math.pow(n , 2); }", tgt_language)
    generated = ""

    while "</s>" not in generated and len(generated) < 10000:
        prediction = predict_next_token(knnmt, translator, src_language, tgt_language, source, generated)
        generated += " " + prediction

    print("Generated", generated)


def predict_next_token(
    knnmt: KNNMT, 
    translator: Translator, 
    src_language: str, 
    tgt_language: str, 
    source: str, 
    generated: str,
):
    decoder_features, targets, target_tokens = translator.get_features(
        input_code=source,
        target_code=generated,
        src_language=src_language,
        tgt_language=tgt_language,
        predict_single_token=True,
        tokenized=True
    )

    language_pair = f"{src_language}_{tgt_language}"
    features = decoder_features[-1]
    knns, distances = knnmt.get_k_nearest_neighbors(features, language_pair)
    tokens = [translator.get_token(id) for id in knns]

    # print("Targets", targets, target_tokens)
    # print("Predictions", knns, tokens, distances)

    return tokens[0]


def add_sample(knnmt: KNNMT):
    src_language = "cpp"
    tgt_language = "java"
    language_pair = f"{src_language}_{tgt_language}"

    src_sample = "static boolean isEven(int n) { return (n % 2) == 0; }"
    tgt_sample = "int isEven(int n) { return (n % 2) == 0; }"

    translator_path = f"models/transcoder_st/Online_ST_{src_language.title()}_{tgt_language.title()}.pth"
    translator = Translator(
        translator_path.replace("Cpp", "CPP"), "data/bpe/cpp-java-python/codes", global_model=True
    )

    # Obtain features and targets from decoder
    decoder_features, targets, target_tokens = translator.get_features(
        input_code=src_sample, 
        target_code=tgt_sample, 
        src_language=src_language, 
        tgt_language=tgt_language,
        tokenized=False
    )

    for i in range(1000):
        knnmt.add_to_datastore(decoder_features, targets, language_pair)
    knnmt.save_datastore(language_pair)
    knnmt.train_datastore(language_pair)


knnmt = KNNMT()
create_datastore(knnmt)
train_datastore(knnmt)
output_sample(knnmt)
# add_sample(knnmt)
# output_sample(knnmt)