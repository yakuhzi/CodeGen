from hashlib import sha256

DATASET_PATH = "dump/transcoder_st"

def combine_dataset():
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

    f = open(f"dump/offline_dataset/cpp_java.cpp.bpe", "a")
    f.writelines(cpp_java_cpp_functions)
    f.close

    f = open(f"dump/offline_dataset/cpp_java.java.bpe", "a")
    f.writelines(cpp_java_java_functions)
    f.close

    f = open(f"dump/offline_dataset/cpp_python.cpp.bpe", "a")
    f.writelines(cpp_python_cpp_functions)
    f.close

    f = open(f"dump/offline_dataset/cpp_python.python.bpe", "a")
    f.writelines(cpp_python_python_functions)
    f.close

    f = open(f"dump/offline_dataset/java_python.java.bpe", "a")
    f.writelines(java_python_java_functions)
    f.close

    f = open(f"dump/offline_dataset/java_python.python.bpe", "a")
    f.writelines(java_python_python_functions)
    f.close

combine_dataset()