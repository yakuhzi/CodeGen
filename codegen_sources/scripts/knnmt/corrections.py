import numpy as np

from .load_functions import extract_functions
from ...model.translate import Translator
from .knnmt import KNNMT

from codegen_sources.model.src.data.dictionary import MASK_WORD
from codegen_sources.model.src.utils import add_noise

DATASET_PATH = "data/test_dataset"
UNSUCCESSFUL_DIR = "codegen_sources/scripts/unsuccessful/all_errors"


def output_sample(knnmt: KNNMT, translator: Translator, language_pair: str, data_index: int):
    src_language, tgt_language = language_pair.split("_")[0], language_pair.split("_")[1]

    # Get tokenized source function
    source_functions = extract_functions(f"{DATASET_PATH}/transcoder_test.{src_language}.tok")
    source = source_functions[data_index]
    source = translator.tokenize(source, src_language)
    generated = ""
    inputs = []

    # Predict target tokens using kNN-MT only
    while "</s>" not in generated and len(generated.split(" ")) < 200:
        prediction, input = predict_next_token(knnmt, translator, src_language, tgt_language, source, generated)
        generated += " " + prediction
        inputs.append(input)

    # Get original TransCoder translation
    translation = translator.translate(source, src_language, tgt_language)[0]
    translator.use_knn_store = False
    original_translation = translator.translate(source, src_language, tgt_language)[0]

    print("Source:", source)
    print("Inputs", inputs)
    print("TC Prediction:", original_translation)
    print("kNN-MT Prediction:", generated)
    print("TC + kNN-MT Prediction:", translation)


def predict_next_token(
    knnmt: KNNMT,
    translator: Translator,
    src_language: str,
    tgt_language: str,
    source: str,
    generated: str
):
    # Get hidden feature representation of last decoder layer and ground truth target tokens
    decoder_features, _, targets, target_tokens, _, _ = translator.get_features(
        input_code=source,
        target_code=generated,
        src_language=src_language,
        tgt_language=tgt_language,
        predict_single_token=True,
        tokenized=True
    )

    # Retrieve k nearest neighbors including their distances and inputs
    language_pair = f"{src_language}_{tgt_language}"
    features = decoder_features[-1].unsqueeze(0)
    knns, distances, inputs = knnmt.get_k_nearest_neighbors(features, language_pair, with_inputs=False)
    tokens = [translator.get_token(id) for id in knns[0]]

    # print("Targets", targets, target_tokens)
    # print("Predictions", knns, tokens, distances)

    return tokens[0], inputs[0][0]


def add_samples(knnmt: KNNMT, translator: Translator, language_pair: str, data_indices: int):
    src_language, tgt_language = language_pair.split("_")[0], language_pair.split("_")[1]

    # Get source and target functions of the TransCoder test dataset
    src_functions = extract_functions(f"{DATASET_PATH}/transcoder_test.{src_language}.tok")
    tgt_functions = extract_functions(f"{DATASET_PATH}/transcoder_test.{tgt_language}.tok")

    for data_index in data_indices:
        # Get source and target sample by the provided indices
        src_sample = src_functions[data_index]
        tgt_sample = tgt_functions[data_index]

        # Tokenize source and target function
        src_tokens = np.array(translator.tokenize(src_sample, src_language).split(" "), dtype="object")
        tgt_tokens = np.array(translator.tokenize(tgt_sample, tgt_language).split(" "), dtype="object")

        src_sample = " ".join(src_tokens)
        tgt_sample = " ".join(tgt_tokens)

        # Obtain features and targets from last decoder layer
        decoder_features, _, targets, target_tokens, input_code, output_code = translator.get_features(
            input_code=src_sample,
            target_code=tgt_sample,
            src_language=src_language,
            tgt_language=tgt_language,
            tokenized=True
        )

        # Add sample to kNN-MT datastore
        knnmt.add_to_datastore(decoder_features, targets, input_code, output_code, language_pair)

    # Save kNN-MT datastore and retrain Faiss index
    knnmt.save_datastore(language_pair)
    knnmt.train_datastore(language_pair)


def get_unsuccessful_indices(src_language: str, tgt_language: str, ):
    # Get IDs of functions that failed in the original TransCoder-ST evaluation
    unsuccessful_path = f"{UNSUCCESSFUL_DIR}/unsuccessful.{src_language}_{tgt_language}"
    unsuccessful_file = open(unsuccessful_path, "r")
    unsuccessful_lines = unsuccessful_file.readlines()
    unsuccessful_file.close()

    # Read lines of TransCoder test dataset
    data_path = f"{DATASET_PATH}/transcoder_test.{tgt_language}.tok"
    data_file = open(data_path, "r")
    data_lines = data_file.readlines()
    data_file.close()

    indices = []

    # Get indices of the unsuccessful IDs in TransCoder test dataset
    for line in unsuccessful_lines:
        function = line.replace("\n", " |")
        index = [idx for idx, s in enumerate(data_lines) if function in s]
        indices.append(index[0])

    return indices


language_pair = "cpp_java"
src_language = language_pair.split("_")[0]
tgt_language = language_pair.split("_")[1]

# Warning: This will modify the datastore provided, as it will add additional keys and values.
# You might want to use a copy of "out/knnmt/mixed".
knnmt = KNNMT("out/knnmt/mixed")

translator_path = f"models/Online_ST_{src_language.title()}_{tgt_language.title()}.pth".replace("Cpp", "CPP")
translator = Translator(
    translator_path,
    "data/bpe/cpp-java-python/codes",
    global_model=True,
    knnmt_dir=knnmt.knnmt_dir
)

# Add all functions that were incorrectly translated by TransCoder-ST to the datastore
indices = get_unsuccessful_indices(src_language, tgt_language)
print("Selected test functions", sorted(indices))
add_samples(knnmt, translator, language_pair, indices)

# Alternative: Add only one sample to the datastore (index in the test dataset, here: 41)
# output_sample(knnmt, translator, language_pair, 41)
# add_samples(knnmt, translator, language_pair, [41])
# output_sample(knnmt, translator, language_pair, 41)
