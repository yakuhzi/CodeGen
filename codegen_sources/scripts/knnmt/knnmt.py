import os
import numpy as np
import faiss
import threading

EMBEDDING_DIMENSION = 1024
# CLUSTERS = 4096
CLUSTERS = 512
PROBE = 32
CODE_SIZE = 64
SEED = 1


class KNNMT:

    def __init__(self, knnmt_dir: str):
        self.lock = threading.Lock()

        self.knnmt_dir = knnmt_dir
        self.faiss_index = {}
        self.datastore_keys = {}
        self.datastore_values = {}
        self.datastore_inputs = {}

        self.datastore_path = os.path.join(knnmt_dir, "datastore")
        self.faiss_index_path = os.path.join(knnmt_dir, "faiss")

    def get_k_nearest_neighbors(self, features, language_pair: str, k: int = 5, with_inputs: bool=False):
        # Load Faiss index and datastore values
        faiss_index = self._load_faiss_index(language_pair)
        datastore_values = self._load_values(language_pair)

        # Get datastore query
        input = features.cpu().detach().numpy().astype('float32')

        # Retrieve kNN-MT datastore results
        distances, knns = faiss_index.search(input, k)
        # Get targets for each retrieved neighbor
        values = [[datastore_values[index] for index in k] for k in knns]

        if not with_inputs:
            return values, distances, [[None for _ in values]]

        # Get the original string translation contexts used to extract the key-value pairs
        datastore_inputs = self._load_inputs(language_pair)
        inputs = [[datastore_inputs[index] for index in k] for k in knns]
        return values, distances, inputs

    def add_to_datastore(self, features, targets, input_code: str, output_code: str, language_pair: str):
        output_code = output_code.split(" ")
        index = 1

        keys = []
        values = []
        inputs = []

        # Add features and targets to the list
        for features, target in zip(features, targets[1:]):
            keys.append(features.detach().cpu().numpy().astype(np.float32))
            values.append(target.cpu().numpy().astype(np.int))
            inputs.append((input_code, ' '.join(output_code[1:index])))
            index += 1

        # Convert to numpy arrays
        keys = np.array(keys, dtype=np.float32)
        values = np.array(values, dtype=np.int)
        inputs = np.array(inputs, dtype=np.str)

        assert keys.shape[0] == values.shape[0] == inputs.shape[0]
        self.lock.acquire()

        # Load existing keys, values and inputs
        datastore_keys = self._load_keys(language_pair)
        datastore_values = self._load_values(language_pair)
        datastore_inputs = self._load_inputs(language_pair)

        # Concatenate keys, values and inputs
        self.datastore_keys[language_pair] = np.concatenate((datastore_keys, keys), axis=0)
        self.datastore_values[language_pair] = np.concatenate((datastore_values, values), axis=0)
        self.datastore_inputs[language_pair] = np.concatenate((datastore_inputs, inputs), axis=0)

        self.lock.release()

    def save_datastore(self, language_pair: str):
        print(f"Saving Datastore for '{language_pair}'")

        # Create datastore directories
        keys_path = f"{self.datastore_path}/keys_{language_pair}.npy"
        values_path = f"{self.datastore_path}/values_{language_pair}.npy"
        inputs_path = f"{self.datastore_path}/inputs_{language_pair}.npy"

        os.makedirs(os.path.dirname(keys_path), exist_ok=True)
        os.makedirs(os.path.dirname(values_path), exist_ok=True)
        os.makedirs(os.path.dirname(inputs_path), exist_ok=True)

        # Get keys, values and inputs for the specified language pair
        datastore_keys = self.datastore_keys[language_pair]
        datastore_values = self.datastore_values[language_pair]
        datastore_inputs = self.datastore_inputs[language_pair]

        # Save datastore
        print("Save Keys:", datastore_keys.shape)
        print("Save Values:", datastore_values.shape)
        print("Save Inputs:", datastore_inputs.shape)

        np.save(keys_path, datastore_keys)
        np.save(values_path, datastore_values)
        np.save(inputs_path, datastore_inputs)

    def train_datastore(self, language_pair):
        print(f"Training Datastore for '{language_pair}'")

        # Load keys and values from datastore
        keys = self._load_keys(language_pair)
        values = self._load_values(language_pair)

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
        faiss_path = f"{self.faiss_index_path}/{language_pair}.faiss"

        if not os.path.exists(faiss_path):
            os.makedirs(os.path.dirname(faiss_path), exist_ok=True)

        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), faiss_path)

    def _load_keys(self, language_pair: str):
        keys_path = f"{self.datastore_path}/keys_{language_pair}.npy"

        if self.datastore_keys.get(language_pair) is not None:
            return self.datastore_keys[language_pair]

        if os.path.exists(keys_path):
            print(f"Loading Datastore Keys for '{language_pair}'")
            datastore_keys = np.load(keys_path)
            self.datastore_keys[language_pair] = datastore_keys
            print("Keys: ", datastore_keys.shape)
        else:
            datastore_keys = np.zeros((0, 1024)).astype('float32')

        return datastore_keys

    def _load_values(self, language_pair: str):
        values_path = f"{self.datastore_path}/values_{language_pair}.npy"

        if self.datastore_values.get(language_pair) is not None:
            return self.datastore_values[language_pair]

        if os.path.exists(values_path):
            print(f"Loading Datastore Values for '{language_pair}'")
            datastore_values = np.load(values_path)
            self.datastore_values[language_pair] = datastore_values
            print("Values: ", datastore_values.shape)
        else:
            datastore_values = np.zeros((0, )).astype('int')

        return datastore_values

    def _load_inputs(self, language_pair: str):
        inputs_path = f"{self.datastore_path}/inputs_{language_pair}.npy"

        if self.datastore_inputs.get(language_pair) is not None:
            return self.datastore_inputs[language_pair]

        if os.path.exists(inputs_path):
            print(f"Loading Datastore Inputs for '{language_pair}'")
            datastore_inputs = np.load(inputs_path)
            self.datastore_inputs[language_pair] = datastore_inputs
            print("Values: ", datastore_inputs.shape)
        else:
            datastore_inputs = np.empty((0, 2)).astype('str')

        return datastore_inputs


    def _load_faiss_index(self, language_pair: str, retrain: bool=False):
        if not retrain and self.faiss_index.get(language_pair) is not None:
            return self.faiss_index[language_pair]

        faiss_path = f"{self.faiss_index_path}/{language_pair}.faiss"

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
