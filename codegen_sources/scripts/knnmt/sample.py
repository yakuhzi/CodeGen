from .load_functions import extract_functions
from ...model.translate import Translator
from .knnmt import KNNMT

DATASET_PATH = "dataset/transcoder/test"


def output_sample(knnmt: KNNMT, language_pair: str, data_index: int):
    src_language, tgt_language = language_pair.split("_")[0], language_pair.split("_")[1]

    translator_path = f"models/transcoder_st/Online_ST_{src_language.title()}_{tgt_language.title()}.pth"
    translator = Translator(
        translator_path.replace("Cpp", "CPP"), "data/bpe/cpp-java-python/codes", global_model=True
    )

    source_functions = extract_functions(f"{DATASET_PATH}/transcoder_test.{src_language}.tok")
    source = translator.tokenize(source_functions[data_index], src_language)
    generated = ""
    inputs = []

    # source = translator.tokenize("int summingSeries ( long n ) { return pow ( n , 2 ) ; }", src_language)
    # file = open("codegen_sources/scripts/test/test.cpp", "r")
    # source = translator.tokenize("".join(file.readlines()), src_language)
    # source = translator.tokenize("float sumOfSeries ( int n ) { return ( 0.666 ) * ( 1 - 1 / pow ( 10 , n ) ) ; }", src_language)

    while "</s>" not in generated and len(generated.split(" ")) < 200:
        prediction, input = predict_next_token(knnmt, translator, src_language, tgt_language, source, generated)
        generated += " " + prediction
        inputs.append(input)

    print("Source:", source)
    print("Generated:", generated)
    print("Inputs", inputs)
    

def predict_next_token(
    knnmt: KNNMT, 
    translator: Translator, 
    src_language: str, 
    tgt_language: str, 
    source: str, 
    generated: str
):
    decoder_features, _, targets, target_tokens, _, _ = translator.get_features(
        input_code=source,
        target_code=generated,
        src_language=src_language,
        tgt_language=tgt_language,
        predict_single_token=True,
        tokenized=True
    )

    language_pair = f"{src_language}_{tgt_language}"
    features = decoder_features[-1].unsqueeze(0)
    knns, distances, inputs = knnmt.get_k_nearest_neighbors(features, language_pair, with_inputs=True)
    tokens = [translator.get_token(id) for id in knns[0]]

    # print("Targets", targets, target_tokens)
    # print("Predictions", knns, tokens, distances)

    return tokens[0], inputs[0][0]

def add_sample(knnmt: KNNMT, language_pair: str, data_index: int):
    src_language, tgt_language = language_pair.split("_")[0], language_pair.split("_")[1]

    src_functions = extract_functions(f"{DATASET_PATH}/transcoder_test.{src_language}.tok")
    tgt_functions = extract_functions(f"{DATASET_PATH}/transcoder_test.{tgt_language}.tok")

    src_sample = src_functions[data_index]
    tgt_sample = tgt_functions[data_index]

    # src_sample = "int summingSeries ( long n ) { return pow ( n , 2 ) ; }"
    # tgt_sample = "static int summingSeries ( long n ) { return ( int ) Math . pow ( n , 2 ) ; }"

    translator_path = f"models/transcoder_st/Online_ST_{src_language.title()}_{tgt_language.title()}.pth"
    translator = Translator(
        translator_path.replace("Cpp", "CPP"), "data/bpe/cpp-java-python/codes", global_model=True
    )

    # Obtain features and targets from decoder
    decoder_features, _, targets, target_tokens = translator.get_features(
        input_code=src_sample, 
        target_code=tgt_sample, 
        src_language=src_language, 
        tgt_language=tgt_language,
        tokenized=False
    )

    knnmt.add_to_datastore(decoder_features, targets, language_pair)
    knnmt.save_datastore(language_pair)
    knnmt.train_datastore(language_pair)

knnmt = KNNMT("/pfs/work7/workspace/scratch/hd_tf268-code-gen/knnmt")

language_pair = "cpp_java"
output_sample(knnmt, language_pair, 229)
# knnmt.train_datastore(language_pair)

# output_sample(knnmt, language_pair, 41)
# add_sample(knnmt, language_pair, 41)
# output_sample(knnmt, language_pair, 41)