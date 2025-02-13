from hashlib import sha256
from typing import Optional


def load_parallel_functions(dataset_path: str, language_pair: str = None):
    parallel_functions = {}

    # Load parallel functions for language pairs containing C++ and Java
    if language_pair is None or language_pair == "cpp_java" or language_pair == "java_cpp":
        cpp_java_cpp_file = open(f"{dataset_path}/cpp_java.cpp.bpe", "r")
        cpp_java_cpp_functions = cpp_java_cpp_file.readlines()

        cpp_java_java_file = open(f"{dataset_path}/cpp_java.java.bpe", "r")
        cpp_java_java_functions = cpp_java_java_file.readlines()

        parallel_functions["cpp_java"] = list(zip(cpp_java_cpp_functions, cpp_java_java_functions))
        parallel_functions["java_cpp"] = list(zip(cpp_java_java_functions, cpp_java_cpp_functions))

    # Load parallel functions for language pairs containing C++ and Python
    if language_pair is None or language_pair == "cpp_python" or language_pair == "python_cpp":
        cpp_python_cpp_file = open(f"{dataset_path}/cpp_python.cpp.bpe", "r")
        cpp_python_cpp_functions = cpp_python_cpp_file.readlines()

        cpp_python_python_file = open(f"{dataset_path}/cpp_python.python.bpe", "r")
        cpp_python_python_functions = cpp_python_python_file.readlines()

        parallel_functions["cpp_python"] = list(zip(cpp_python_cpp_functions, cpp_python_python_functions))
        parallel_functions["python_cpp"] = list(zip(cpp_python_python_functions, cpp_python_cpp_functions))

    # Load parallel functions for language pairs containing Java and Python
    if language_pair is None or language_pair == "java_python" or language_pair == "python_java":
        java_python_java_file = open(f"{dataset_path}/java_python.java.bpe", "r")
        java_python_java_functions = java_python_java_file.readlines()

        java_python_python_file = open(f"{dataset_path}/java_python.python.bpe", "r")
        java_python_python_functions = java_python_python_file.readlines()

        parallel_functions["java_python"] = list(zip(java_python_java_functions, java_python_python_functions))
        parallel_functions["python_java"] = list(zip(java_python_python_functions, java_python_java_functions))

    # Deduplicate parallel functions
    parallel_functions = deduped_parallel_functions(parallel_functions)

    if language_pair is not None:
        return parallel_functions[language_pair]

    return parallel_functions


def load_validation_functions(validation_set_path: str, language_pair: str = None, half: Optional[int] = None):
    # Load C++ functions
    if language_pair is None or "cpp" in language_pair:
        cpp_functions = extract_functions(f"{validation_set_path}/transcoder_valid.cpp.tok")

    # Load Java functions
    if language_pair is None or "java" in language_pair:
        java_functions = extract_functions(f"{validation_set_path}/transcoder_valid.java.tok")

    # Load Python functions
    if language_pair is None or "python" in language_pair:
        python_functions = extract_functions(f"{validation_set_path}/transcoder_valid.python.tok")

    parallel_functions = {}

    # Combine functions to obtain parallel functions
    if language_pair is None or language_pair == "cpp_java":
        parallel_functions["cpp_java"] = list(zip(cpp_functions, java_functions))

    if language_pair is None or language_pair == "cpp_python":
        parallel_functions["cpp_python"] = list(zip(cpp_functions, python_functions))

    if language_pair is None or language_pair == "java_cpp":
        parallel_functions["java_cpp"] = list(zip(java_functions, cpp_functions))

    if language_pair is None or language_pair == "java_python":
        parallel_functions["java_python"] = list(zip(java_functions, python_functions))

    if language_pair is None or language_pair == "python_cpp":
        parallel_functions["python_cpp"] = list(zip(python_functions, cpp_functions))

    if language_pair is None or language_pair == "python_java":
        parallel_functions["python_java"] = list(zip(python_functions, java_functions))

    if language_pair is not None:
        if half is None:
            # Return parallel function for language pair
            return parallel_functions[language_pair]
        elif half == 1:
            # Return first half of parallel functions
            return parallel_functions[language_pair][:int(len(parallel_functions[language_pair]) / 2)]
        else:
            # Return second half of parallel functions
            return parallel_functions[language_pair][int(len(parallel_functions[language_pair]) / 2):]

    return parallel_functions


def extract_functions(path):
    def extract_function(line):
        return " | ".join(line.split(" | ")[1:])

    file = open(path, "r")
    return [extract_function(line) for line in file.readlines()]


def deduped_parallel_functions(parallel_functions):
    deduped_functions = {}

    for language_pair in parallel_functions.keys():
        # Get parallel functions of language pair
        function_pairs = parallel_functions[language_pair]
        deduped_pairs = []
        hashes = {}

        for pair in function_pairs:
            source, target = pair

            # Skip empty lines
            if source == "\n" or target == "\n":
                continue

            # Hash source and target function
            hash = sha256(source.encode("utf8") + target.encode("utf8")).hexdigest()

            # If hash has not been generated before, add to deduplicated list
            if hashes.get(hash) is None:
                hashes[hash] = pair
                deduped_pairs.append(pair)

        deduped_functions[language_pair] = deduped_pairs

        print(f"Size of '{language_pair}' dataset: {len(function_pairs)}")
        print(f"Size of deduped '{language_pair}' dataset: {len(deduped_pairs)}")

    return deduped_functions
