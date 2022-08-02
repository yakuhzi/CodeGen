import os
import faiss
from tqdm import tqdm
import numpy as np

from ...model.translate import Translator

LANGUAGE_PAIRS = [
    "cpp_java",
    "cpp_python",
    "java_cpp",
    "java_python",
    "python_cpp",
    "python_java",
]

DATASET_PATH = "dump/transcoder_st/dataset_01/offline_dataset"
DATASTORE_PATH = "dump/transcoder_st_knnmt/datastore"
FAISS_INDEX_PATH = "dump/transcoder_st_knnmt/faiss"
TREE_SITTER_ROOT = "tree-sitter"
EMBEDDING_DIMENSION = 1024
CLUSTERS = 4096
CODE_SIZE = 64
PROBE = 32
SEED = 1
NUM_KEYS_PER_ITERATION = 500000


def create_datastore():
    cpp_java_cpp_file = open(f"{DATASET_PATH}/train.cpp_sa-java_sa.cpp_sa.bpe", "r")
    cpp_java_cpp_functions = cpp_java_cpp_file.readlines()

    cpp_java_java_file = open(f"{DATASET_PATH}/train.cpp_sa-java_sa.java_sa.bpe", "r")
    cpp_java_java_functions = cpp_java_java_file.readlines()

    cpp_python_cpp_file = open(f"{DATASET_PATH}/train.cpp_sa-python_sa.cpp_sa.bpe", "r")
    cpp_python_cpp_functions = cpp_python_cpp_file.readlines()

    cpp_python_python_file = open(f"{DATASET_PATH}/train.cpp_sa-python_sa.python_sa.bpe", "r")
    cpp_python_python_functions = cpp_python_python_file.readlines()

    java_python_java_file = open(f"{DATASET_PATH}/train.java_sa-python_sa.java_sa.bpe", "r")
    java_python_java_functions = java_python_java_file.readlines()

    java_python_python_file = open(f"{DATASET_PATH}/train.java_sa-python_sa.python_sa.bpe", "r")
    java_python_python_functions = java_python_python_file.readlines()

    parallel_functions = {}
    parallel_functions["cpp_java"] = zip(cpp_java_cpp_functions, cpp_java_java_functions)
    parallel_functions["cpp_python"] = zip(cpp_python_cpp_functions, cpp_python_python_functions)
    parallel_functions["java_cpp"] = zip(cpp_java_java_functions, cpp_java_cpp_functions)
    parallel_functions["java_python"] = zip(java_python_java_functions, java_python_python_functions)
    parallel_functions["python_cpp"] = zip(cpp_python_python_functions, cpp_python_cpp_functions)
    parallel_functions["python_java"] = zip(java_python_python_functions, java_python_java_functions)

    for language_pair in LANGUAGE_PAIRS:
        print("#" * 10 + f" Creating Datastore for '{language_pair}' " + "#" * 10)

        keys_path = f"{DATASTORE_PATH}/keys_{language_pair}.npy"
        values_path = f"{DATASTORE_PATH}/values_{language_pair}.npy"

        src_language = language_pair.split("_")[0]
        tgt_language = language_pair.split("_")[1]

        translator_path = f"models/transcoder_st/Online_ST_{src_language.title()}_{tgt_language.title()}.pth"
        translator = Translator(
            translator_path.replace("Cpp", "CPP"), "data/bpe/cpp-java-python/codes", global_model=True
        )

        keys = []
        values = []

        # src = "int mult ( int x , int y ) { return x * y ; }"
        # tgt = "public static int mult ( int x , int y ) { return x * y ; }"

        # Obtain features and targets from decoder
        # for i in range(10):
        for src_sample, tgt_sample in tqdm([(src, tgt)]):
            decoder_features, targets, target_tokens = translator.get_features(
                input_code=src_sample, target_code=tgt_sample, src_lang=src_language, tgt_lang=tgt_language,
            )

            for features, target, token in zip(decoder_features, targets[1:], target_tokens[1:]):
                keys.append(features.detach().cpu().numpy().astype(np.float32))
                values.append(target.cpu().numpy().astype(np.int))

        keys = np.array(keys, dtype=np.float32)
        values = np.array(values, dtype=np.int)

        # Load keys datastore
        if os.path.exists(keys_path):
            datastore_keys = np.load(keys_path)
            print("keys1", datastore_keys.shape)
            datastore_keys = np.concatenate((datastore_keys, keys), axis=0)
        else:
            os.makedirs(os.path.dirname(keys_path), exist_ok=True)
            datastore_keys = keys

        # Load values datastore
        if os.path.exists(values_path):
            datastore_values = np.load(values_path)
            print("vals1", datastore_values.shape)
            datastore_values = np.concatenate((datastore_values, values), axis=0)
        else:
            os.makedirs(os.path.dirname(values_path), exist_ok=True)
            datastore_values = values

        print("keys2", datastore_keys.shape)
        print("vals2", datastore_values.shape)

        # Save datastore
        np.save(keys_path, datastore_keys)
        np.save(values_path, datastore_values)


def train_datastore():
    # Initialize faiss index
    resources = faiss.StandardGpuResources()
    options = faiss.GpuClonerOptions()
    options.useFloat16 = True

    for language_pair in LANGUAGE_PAIRS:
        # Load keys and values from datastore
        keys_path = f"{DATASTORE_PATH}/keys_{language_pair}.npy"
        values_path = f"{DATASTORE_PATH}/values_{language_pair}.npy"

        keys = np.load(keys_path)
        values = np.load(values_path)

        print("SIZEEEK", len(keys), keys.shape)
        print("SIZEEE", len(values), values.shape)

        # Initialize faiss
        quantizer = faiss.IndexFlatL2(EMBEDDING_DIMENSION)

        CLUSTERS = 10
        PROBE = 2

        index = faiss.IndexIVFPQ(quantizer, EMBEDDING_DIMENSION, CLUSTERS, CODE_SIZE, 8)
        index.nprobe = PROBE
        gpu_index = faiss.index_cpu_to_gpu(resources, 0, index, options)

        # Training faiss index
        print("#" * 10 + f" Training Index for '{language_pair}' " + "#" * 10)
        np.random.seed(SEED)
        random_sample = np.random.choice(
            np.arange(values.shape[0]), size=[min(1000000, values.shape[0])], replace=False
        )
        gpu_index.train(keys[random_sample].astype(np.float32))

        # Adding keys to index
        print(f"#" * 10 + f" Adding keys for '{language_pair}' " + "#" * 10)
        gpu_index.add_with_ids(keys.astype(np.float32), np.arange(keys.shape[0]))

        # Write faiss index
        faiss_path = f"{FAISS_INDEX_PATH}/{language_pair}.faiss"

        if not os.path.exists(faiss_path):
            os.makedirs(os.path.dirname(faiss_path), exist_ok=True)

        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), faiss_path)


def get_k_nearest_neighbors(input, src_language: str, tgt_language: str, k: int = 5):
    language_pair = f"{src_language}_{tgt_language}"
    faiss_path = f"{FAISS_INDEX_PATH}/{language_pair}.faiss"

    index = faiss.read_index(faiss_path, faiss.IO_FLAG_ONDISK_SAME_DIR)
    resources = faiss.StandardGpuResources()
    options = faiss.GpuClonerOptions()
    options.useFloat16 = True

    index = faiss.index_cpu_to_gpu(resources, 0, index, options)
    distances, knns = index.search(input, k)
    return knns, distances

def get_sample():
    src_language = "cpp"
    tgt_language = "java"
    language_pair = f"{src_language}_{tgt_language}"

    source = "int mult ( int x , int y ) { return x * y ; }"
    generated = "public static int mult ( int x , int y ) { return x *"

    translator_path = f"models/transcoder_st/Online_ST_{src_language.title()}_{tgt_language.title()}.pth"
    translator = Translator(
        translator_path.replace("Cpp", "CPP"), "data/bpe/cpp-java-python/codes", global_model=True
    )

    features, targets, target_tokens = translator.get_features(
        input_code=source, 
        target_code=generated, 
        src_lang=src_language, 
        tgt_lang=tgt_language, 
        predict_single_token=True
    )

    query = np.zeros((1, 1024)).astype('float32')
    query[0] = features[-1].cpu()

    knns, distances = get_k_nearest_neighbors(query, src_language="cpp", tgt_language="java")
    values_path = f"{DATASTORE_PATH}/values_{language_pair}.npy"
    datastore_values = np.load(values_path)

    values = [datastore_values[index] for index in knns[0]]
    tokens = get_tokens(values)
    print("T", targets, target_tokens)
    print("P", values, tokens)

def get_tokens(ids, src_language="cpp", tgt_language="java"):
    translator_path = f"models/transcoder_st/Online_ST_{src_language.title()}_{tgt_language.title()}.pth"
    translator = Translator(
        translator_path.replace("Cpp", "CPP"), "data/bpe/cpp-java-python/codes", global_model=True
    )

    return [translator.get_token(id) for id in ids]

# create_datastore()
# train_datastore()
get_sample()
