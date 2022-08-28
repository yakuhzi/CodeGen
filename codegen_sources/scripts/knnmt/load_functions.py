from hashlib import sha256

DATASET_PATH = "dump/offline_dataset"
VALIDATION_SET_PATH = "dataset/transcoder/test"


def load_parallel_functions(language_pair: str=None):
    parallel_functions = {}

    if language_pair is None or language_pair == "cpp_java" or language_pair == "java_cpp":
        cpp_java_cpp_file = open(f"{DATASET_PATH}/cpp_java.cpp.bpe", "r")
        cpp_java_cpp_functions = cpp_java_cpp_file.readlines()

        cpp_java_java_file = open(f"{DATASET_PATH}/cpp_java.java.bpe", "r")
        cpp_java_java_functions = cpp_java_java_file.readlines()

        parallel_functions["cpp_java"] = list(zip(cpp_java_cpp_functions, cpp_java_java_functions))
        parallel_functions["java_cpp"] = list(zip(cpp_java_java_functions, cpp_java_cpp_functions))

    if language_pair is None or language_pair == "cpp_python" or language_pair == "python_cpp":
        cpp_python_cpp_file = open(f"{DATASET_PATH}/cpp_python.cpp.bpe", "r")
        cpp_python_cpp_functions = cpp_python_cpp_file.readlines()

        cpp_python_python_file = open(f"{DATASET_PATH}/cpp_python.python.bpe", "r")
        cpp_python_python_functions = cpp_python_python_file.readlines()

        parallel_functions["cpp_python"] = list(zip(cpp_python_cpp_functions, cpp_python_python_functions))
        parallel_functions["python_cpp"] = list(zip(cpp_python_python_functions, cpp_python_cpp_functions))

    if language_pair is None or language_pair == "java_python" or language_pair == "python_java":
        java_python_java_file = open(f"{DATASET_PATH}/java_python.java.bpe", "r")
        java_python_java_functions = java_python_java_file.readlines()

        java_python_python_file = open(f"{DATASET_PATH}/java_python.python.bpe", "r")
        java_python_python_functions = java_python_python_file.readlines()

        parallel_functions["java_python"] = list(zip(java_python_java_functions, java_python_python_functions))
        parallel_functions["python_java"] = list(zip(java_python_python_functions, java_python_java_functions))

    parallel_functions = deduped_parallel_functions(parallel_functions)

    if language_pair is not None:
        return parallel_functions[language_pair]

    return parallel_functions

def load_validation_functions():
    cpp_functions = extract_functions(f"{VALIDATION_SET_PATH}/transcoder_valid.cpp.tok")
    java_functions = extract_functions(f"{VALIDATION_SET_PATH}/transcoder_valid.java.tok")
    python_functions = extract_functions(f"{VALIDATION_SET_PATH}/transcoder_valid.python.tok")

    parallel_functions = {}
    parallel_functions["cpp_java"] = list(zip(cpp_functions, java_functions))
    parallel_functions["cpp_python"] = list(zip(cpp_functions, python_functions))
    parallel_functions["java_cpp"] = list(zip(java_functions, cpp_functions))
    parallel_functions["java_python"] = list(zip(java_functions, python_functions))
    parallel_functions["python_cpp"] = list(zip(python_functions, cpp_functions))
    parallel_functions["python_java"] = list(zip(python_functions, java_functions))
    return parallel_functions


def extract_functions(path):
    def extract_function(line):
        return " | ".join(line.split(" | ")[1:])

    file = open(path, "r")
    return [extract_function(line) for line in file.readlines()]


def deduped_parallel_functions(parallel_functions):
    deduped_functions = {}

    for language_pair in parallel_functions.keys():
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