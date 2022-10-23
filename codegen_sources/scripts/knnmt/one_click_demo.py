import numpy as np

from .load_functions import extract_functions
from ...model.translate import Translator
from .knnmt import KNNMT

DATASET_PATH = "data/test_dataset"


def output_sample(knnmt: KNNMT, translator: Translator, language_pair: str, data_index: int):
    src_language, tgt_language = language_pair.split("_")[0], language_pair.split("_")[1]

    # Get tokenized source function
    source_functions = extract_functions(f"{DATASET_PATH}/transcoder_test.{src_language}.tok")
    source = source_functions[data_index]
    source = translator.tokenize(source, src_language)
    generated = ""
    inputs = ""

    # Predict target tokens using kNN-MT only
    while "</s>" not in generated and len(generated.split(" ")) < 200:
        prediction, input = predict_next_token(knnmt, translator, src_language, tgt_language, source, generated)
        generated += " " + prediction
        inputs += f"{input[0]}\n{input[1]}\n"

    # Get original TransCoder translation
    translation = translator.translate(source, src_language, tgt_language)[0]
    translator.use_knn_store = False
    original_translation = translator.translate(source, src_language, tgt_language)[0]
    
    target_functions = extract_functions(f"{DATASET_PATH}/transcoder_test.{tgt_language}.tok")
    target = target_functions[data_index]
    target = translator.tokenize(target, tgt_language)
    
    print("\n\n\n\n\n")
    print(f"FINAL PREDICTION: '{generated[1:]}'")
    print(f"GROUND TRUTH: '{target} </s>'")
    print("\n\n")


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
    knns, distances, inputs = knnmt.get_k_nearest_neighbors(features, language_pair, with_inputs=True)
    tokens = [translator.get_token(id) for id in knns[0]]
    
    print("\n\n\n\n\n")
    print("=" * 100)
    print(f"SOURCE: '{source}'")
    print(f"GENERATED: '{generated[1:]}'\n")
    print("-" * 100)
    print(f"\nNEXT TC TARGET: '{target_tokens[-1]}'")
    print(f"PREDICTIONS: {tokens}")
    print(f"DISTANCES: {distances[0].astype(int)}\n")
    print("-" * 100)
    print(f"\nINPUT SOURCE: '{inputs[0][0][0][5:]}'")
    print(f"INPUT TARGET: '{inputs[0][0][1]}'")
    print("=" * 100)

    # import pdb; pdb.set_trace()
    return tokens[0], inputs[0][0]


language_pair = "cpp_java"
src_language = language_pair.split("_")[0]
tgt_language = language_pair.split("_")[1]

knnmt = KNNMT("out/knnmt/live_demo")

translator_path = f"models/Online_ST_{src_language.title()}_{tgt_language.title()}.pth".replace("Cpp", "CPP")
translator = Translator(
    translator_path,
    "data/bpe/cpp-java-python/codes",
    global_model=True,
    knnmt_dir=knnmt.knnmt_dir
)

output_sample(knnmt, translator, language_pair, 442)
