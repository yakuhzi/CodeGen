import os
import numpy as np
import faiss

DATASTORE_PATH = "dump/transcoder_st_knnmt/datastore"
FAISS_INDEX_PATH = "dump/transcoder_st_knnmt/faiss"
EMBEDDING_DIMENSION = 1024
# CLUSTERS = 4096
CLUSTERS = 512
PROBE = 32
CODE_SIZE = 64
SEED = 1


class KNNMT:

    def __init__(self):
        self.faiss_index = {}
        self.datastore_keys = {}
        self.datastore_values = {}
        self.new_examples = []


    def add_to_datastore(self, decoder_features, targets, language_pair: str):
        keys = []
        values = []

        for features, target in zip(decoder_features, targets[1:]):
            key = features.detach().cpu().numpy().astype(np.float32)
            keys.append(key)
            self.new_examples.append(key)
            values.append(target.cpu().numpy().astype(np.int))

        keys = np.array(keys, dtype=np.float32)
        values = np.array(values, dtype=np.int)

        datastore_keys = self._load_keys(language_pair)
        datastore_values = self._load_values(language_pair)

        self.datastore_keys[language_pair] = np.concatenate((datastore_keys, keys), axis=0)
        self.datastore_values[language_pair] = np.concatenate((datastore_values, values), axis=0)


    def save_datastore(self, language_pair: str):
        print(f"Saving Datastore for '{language_pair}'")

        keys_path = f"{DATASTORE_PATH}/keys_{language_pair}.npy"
        values_path = f"{DATASTORE_PATH}/values_{language_pair}.npy"

        os.makedirs(os.path.dirname(keys_path), exist_ok=True)
        os.makedirs(os.path.dirname(values_path), exist_ok=True)

        datastore_keys = self.datastore_keys[language_pair]
        datastore_values = self.datastore_values[language_pair]
        
        # Save datastore
        np.save(keys_path, datastore_keys)
        np.save(values_path, datastore_values)

        
    def train_datastore(self, language_pair):
        print(f"Training Datastore for '{language_pair}'")

        # Initialize faiss index
        resources = faiss.StandardGpuResources()
        options = faiss.GpuClonerOptions()
        options.useFloat16 = True

        # Load keys and values from datastore
        keys = self._load_keys(language_pair)
        values = self._load_values(language_pair)
        print(keys.shape)

        # Initialize faiss
        gpu_index = self._load_faiss_index(language_pair, retrain=True)

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


    def get_k_nearest_neighbors(self, features, language_pair: str, k: int = 5):
        faiss_index = self._load_faiss_index(language_pair)
        datastore_values = self._load_values(language_pair)

        input = np.zeros((1, 1024)).astype('float32')
        input[0] = features.cpu()

        distances, knns = faiss_index.search(input, k)

        values = [datastore_values[index] for index in knns[0]]
        distances = distances[0]
        return values, distances


    def _load_keys(self, language_pair: str):
        keys_path = f"{DATASTORE_PATH}/keys_{language_pair}.npy"

        if self.datastore_keys.get(language_pair) is not None:
            return self.datastore_keys[language_pair]

        if os.path.exists(keys_path):
            print(f"Loading Datastore Keys for '{language_pair}'")
            datastore_keys = np.load(keys_path)
            self.datastore_keys[language_pair] = datastore_keys
        else:
            datastore_keys = np.zeros((0, 1024)).astype('float32')

        return datastore_keys


    def _load_values(self, language_pair: str):
        values_path = f"{DATASTORE_PATH}/values_{language_pair}.npy"

        if self.datastore_values.get(language_pair) is not None:
            return self.datastore_values[language_pair]

        if os.path.exists(values_path):
            print(f"Loading Datastore Values for '{language_pair}'")
            datastore_values = np.load(values_path)
            self.datastore_values[language_pair] = datastore_values
        else:
            datastore_values = np.zeros((0, )).astype('int')

        return datastore_values


    def _load_faiss_index(self, language_pair: str, retrain: bool=False):
        if not retrain and self.faiss_index.get(language_pair) is not None:
            return self.faiss_index[language_pair]

        faiss_path = f"{FAISS_INDEX_PATH}/{language_pair}.faiss"

        if not retrain and os.path.exists(faiss_path):
            print(f"Loading Faiss Index for '{language_pair}'")
            index = faiss.read_index(faiss_path, faiss.IO_FLAG_ONDISK_SAME_DIR)
        else:
            quantizer = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
            index = faiss.IndexIVFPQ(quantizer, EMBEDDING_DIMENSION, CLUSTERS, CODE_SIZE, 8)
            index.nprobe = PROBE

        resources = faiss.StandardGpuResources()
        options = faiss.GpuClonerOptions()
        options.useFloat16 = True
            
        gpu_index = faiss.index_cpu_to_gpu(resources, 0, index, options)
        self.faiss_index[language_pair] = gpu_index
        return gpu_index
