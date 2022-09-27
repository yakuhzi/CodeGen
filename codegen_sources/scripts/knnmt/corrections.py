import numpy as np

from .load_functions import extract_functions
from ...model.translate import Translator
from .knnmt import KNNMT

from codegen_sources.model.src.data.dictionary import MASK_WORD
from codegen_sources.model.src.utils import add_noise

DATASET_PATH = "data/test_dataset"


def output_sample(knnmt: KNNMT, translator: Translator, language_pair: str, data_index: int):
    src_language, tgt_language = language_pair.split("_")[0], language_pair.split("_")[1]

    source_functions = extract_functions(f"{DATASET_PATH}/transcoder_test.{src_language}.tok")
    source = source_functions[data_index]
    # source = translator.tokenize(source, src_language)
    generated = ""
    inputs = []

    # source = translator.tokenize("int summingSeries ( long n ) { return pow ( n , 2 ) ; }", src_language)
    # file = open("codegen_sources/scripts/test/test.cpp", "r")
    # source = translator.tokenize("".join(file.readlines()), src_language)
    # source = translator.tokenize("float sumOfSeries ( int n ) { return ( 0.666 ) * ( 1 - 1 / pow ( 10 , n ) ) ; }", src_language)

    while "</s>" not in generated and len(generated.split(" ")) < 200:
        prediction = predict_next_token(knnmt, translator, src_language, tgt_language, source, generated)
        generated += " " + prediction
        # inputs.append(input)

    translation = translator.translate(source, src_language, tgt_language)[0]
    translator.use_knn_store = False
    original_translation = translator.translate(source, src_language, tgt_language)[0]

    print("Source:", source)
    # print("Inputs", inputs)
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
    knns, distances = knnmt.get_k_nearest_neighbors(features, language_pair, with_inputs=False)
    tokens = [translator.get_token(id) for id in knns[0]]

    # print("Targets", targets, target_tokens)
    # print("Predictions", knns, tokens, distances)

    # return tokens[0], inputs[0][0]
    return tokens[0]

def add_samples(knnmt: KNNMT, translator: Translator, language_pair: str, data_indices: int):
    src_language, tgt_language = language_pair.split("_")[0], language_pair.split("_")[1]

    src_functions = extract_functions(f"{DATASET_PATH}/transcoder_test.{src_language}.tok")
    tgt_functions = extract_functions(f"{DATASET_PATH}/transcoder_test.{tgt_language}.tok")

    for data_index in data_indices:
        src_sample = src_functions[data_index]
        tgt_sample = tgt_functions[data_index]

        src_tokens = np.array(translator.tokenize(src_sample, src_language).split(" "), dtype="object")
        tgt_tokens = np.array(translator.tokenize(tgt_sample, tgt_language).split(" "), dtype="object")

        randomState = np.random.RandomState(2022)

        # # x1, len1 = translator.get_x1_len1(src_sample, src_language)
        # # print(x1, len1)

        # # params = {
        # #     "word_shuffle": 0.2,
        # #     "word_dropout": 0.2,
        # #     "mask_length_dist": "",
        # #     "pad_index": "",
        # #     "eos_index": "",
        # #     "word_mask_keep_rand": "0.8,0.1,0.1"
        # # }

        # # parser = get_parser()
        # # params = parser.parse_args()

        # # # x1, len1 = add_noise(x1, len1, params, 64000, randomState)
        # # # print(x1, len1)

        # # data = load_data(params)
        # # model = build_model(params, data["dico"])

        # # trainer = SingleTrainer(model=model, data=None, params=params)
        # # x, y, pred_mask = trainer.mask_out(x1, len1)
        # # print(x, y, pred_mask)

        # src_indices = randomState.choice(len(src_tokens), int(len(src_tokens) * 0.10), replace=False) 
        # tgt_indices = randomState.choice(len(tgt_tokens), int(len(tgt_tokens) * 0.10), replace=False) 

        # src_chunks = np.array_split(src_indices, 2)
        # tgt_chunks = np.array_split(tgt_indices, 2)

        # np.put(src_tokens, src_chunks[1], [MASK_WORD for _ in src_chunks[1]])
        # np.put(tgt_tokens, tgt_chunks[1], [MASK_WORD for _ in tgt_chunks[1]])

        # src_tokens = np.delete(src_tokens, src_chunks[0])
        # tgt_tokens = np.delete(tgt_tokens, tgt_chunks[0])

        src_sample = " ".join(src_tokens)
        tgt_sample = " ".join(tgt_tokens)

        # Obtain features and targets from decoder
        decoder_features, _, targets, target_tokens, input_code, output_code = translator.get_features(
            input_code=src_sample, 
            target_code=tgt_sample, 
            src_language=src_language, 
            tgt_language=tgt_language,
            tokenized=True
        )

        knnmt.add_to_datastore(decoder_features, targets, input_code, output_code, language_pair)

    #knnmt.save_datastore(language_pair)
    knnmt.train_datastore(language_pair)

def get_unsuccessful_indices(src_language: str, tgt_language: str, ):
    UNSUCCESSFUL_DIR = "codegen_sources/scripts/unsuccessful/all_errors"
    unsuccessful_path = f"{UNSUCCESSFUL_DIR}/unsuccessful.{src_language}_{tgt_language}"
    unsuccessful_file = open(unsuccessful_path, "r")
    unsuccessful_lines = unsuccessful_file.readlines()
    unsuccessful_file.close()

    data_path = f"{DATASET_PATH}/transcoder_test.{tgt_language}.tok"
    data_file = open(data_path, "r")
    data_lines = data_file.readlines()
    data_file.close()

    indices = []

    for line in unsuccessful_lines:
        function = line.replace("\n", " |")
        index = [idx for idx, s in enumerate(data_lines) if function in s]
        indices.append(index[0])

    return indices
    

language_pair = "cpp_java"
src_language = language_pair.split("_")[0]
tgt_language = language_pair.split("_")[1]

knnmt = KNNMT("out/knnmt/sample")

translator_path = f"models/Online_ST_{src_language.title()}_{tgt_language.title()}.pth".replace("Cpp", "CPP")
translator = Translator(
    translator_path, 
    "data/bpe/cpp-java-python/codes", 
    global_model=True,
    knnmt_dir=knnmt.knnmt_dir
)

indices = get_unsuccessful_indices(src_language, tgt_language)
print("Selected test functions", sorted(indices))
# add_samples(knnmt, translator, language_pair, indices)

knnmt.train_datastore(language_pair)
output_sample(knnmt, translator, language_pair, 41)
add_samples(knnmt, translator, language_pair, [41])
output_sample(knnmt, translator, language_pair, 41)