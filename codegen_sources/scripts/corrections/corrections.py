import argparse
import os
from pathlib import Path
from typing import Tuple

import torch

from ...model.src.evaluation.comp_acc_computation import (run_cpp_program,
                                                          run_java_program,
                                                          run_python_program)
from ...model.src.utils import get_errors
from ...model.translate import Translator
from ...preprocessing.lang_processors.lang_processor import LangProcessor
from .fix_code import fix_code


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="TransCoder Corrections")

    # main parameters
    parser.add_argument(
        "--src_lang", "-s", type=str, default="cpp", help="Source language"
    )

    parser.add_argument(
        "--tgt_lang", "-t", type=str, default="java", help="Target language"
    )

    return parser

def run_program(script_path: str):
    if TGT_LANGUAGE == "cpp":
        return run_cpp_program(script_path, 0)
    elif TGT_LANGUAGE == "java":
        return run_java_program(script_path, 0)
    elif TGT_LANGUAGE == "python":
        return run_python_program(script_path, 0)


parser = get_parser()
params = parser.parse_args()

TREE_SITTER_ROOT = Path(__file__).parents[3].joinpath("tree-sitter")
SRC_LANGUAGE = params.src_lang
TGT_LANGUAGE = params.tgt_lang
RUN = os.listdir(f"dump/transcoder_st/eval/{SRC_LANGUAGE}_{TGT_LANGUAGE}/online_st")[0]
RUN_PATH = f"dump/transcoder_st/eval/{SRC_LANGUAGE}_{TGT_LANGUAGE}/online_st/{RUN}"
PATH = f'dataset/transcoder/test/transcoder_test.{SRC_LANGUAGE}.tok'
OUT_DIR = f'dump/transcoder_st_corrections/{SRC_LANGUAGE}_{TGT_LANGUAGE}'
FILLED_OUT_DIR = f"{OUT_DIR}/filled"
FIXED_OUT_DIR = f"{OUT_DIR}/fixed"
COMPILE_OUT_DIR = f"{OUT_DIR}/compile"
USE_KNN_STORE = False

src_lang_processor = LangProcessor.processors[SRC_LANGUAGE](root_folder=TREE_SITTER_ROOT)
tgt_lang_processor = LangProcessor.processors[TGT_LANGUAGE](root_folder=TREE_SITTER_ROOT)

print("=" * 100)
print(f"Corrections for {SRC_LANGUAGE} -> {TGT_LANGUAGE}")
print("=" * 100)

with torch.no_grad():
    translator_path = f"models/transcoder_st/Online_ST_{SRC_LANGUAGE.title()}_{TGT_LANGUAGE.title()}.pth"
    translator = Translator(translator_path.replace("Cpp", "CPP"), 'data/bpe/cpp-java-python/codes')

    if not os.path.exists(FILLED_OUT_DIR):
        os.makedirs(FILLED_OUT_DIR)

    if not os.path.exists(FIXED_OUT_DIR):
        os.makedirs(FIXED_OUT_DIR)

    with open(PATH) as file:
        total = 0
        skipped = 0
        executed = 0
        success_1 = 0
        success_2 = 0
        fixed = 0
        broken = 0

        for index, line in enumerate(file):
            function_id = line.split(" | ")[0]
            file_suffix = "py" if TGT_LANGUAGE == "python" else TGT_LANGUAGE
            original_eval_path = f"data/transcoder_evaluation_gfg/{TGT_LANGUAGE}/{function_id}.{file_suffix}"
            filled_eval_path = f"{RUN_PATH}/eval_scripts/{SRC_LANGUAGE}_sa-{TGT_LANGUAGE}_sa.test/{function_id}.{file_suffix}"

            total += 1

            whitelist = [
                "MAXIMUM_AREA_RECTANGLE_PICKING_FOUR_SIDES_ARRAY",
                # "SUM_AREA_RECTANGLES_POSSIBLE_ARRAY",
                "PARTITION_INTO_TWO_SUBARRAYS_OF_LENGTHS_K_AND_N_K_SUCH_THAT_THE_DIFFERENCE_OF_SUMS_IS_MAXIMUM"
            ]
            
            if function_id not in whitelist:
                pass

            if not os.path.exists(filled_eval_path):
                skipped += 1
                continue

            print("+" * 100)
            print(f"Executing '{function_id}'", index)
            print("+" * 100)
            executed += 1

            function = " | ".join(line.split(" | ")[1:])
            function = src_lang_processor.detokenize_code(function)

            filled_script_model = open(filled_eval_path, "r").read()
            original_script_model = open(original_eval_path, "r").read()

            filled_script_path = f"{FILLED_OUT_DIR}/{function_id}.{file_suffix}"
            fixed_script_path = f"{FIXED_OUT_DIR}/{function_id}.{file_suffix}"

            open(filled_script_path, "w", encoding="utf-8").write(filled_script_model)
            result_1, _ = run_program(filled_script_path)

            if TGT_LANGUAGE == "python":
                original_script_model = f"import numpy as np \nimport math\nfrom math import *\nimport collections\nfrom collections import *\nimport heapq\nimport itertools\nimport random\nimport sys\n\n{original_script_model}"

            f_fill = output = translator.translate(
                function,
                lang1=SRC_LANGUAGE,
                lang2=TGT_LANGUAGE,
                beam_size=1,
                use_knn_store=USE_KNN_STORE
            )[0].replace("@ @", "")

            f_name = tgt_lang_processor.get_function_name(f_fill)
            errors = get_errors(f_fill, TGT_LANGUAGE)

            fixed_script_model = fix_code(
                original_script_model,
                f_fill, 
                TGT_LANGUAGE,
                tgt_lang_processor,
                f_name,
                errors
            )

            open(fixed_script_path, "w", encoding="utf-8").write(fixed_script_model)
            result_2, _ = run_program(fixed_script_path)

            if result_1[0] == "success":
                success_1 += 1
            if result_2[0] == "success":
                success_2 += 1

            if result_1[0] != "success" and result_2[0] == "success":
                fixed += 1
            elif result_1[0] == "success" and result_2[0] != "success":
                broken += 1

            print("Result: ", result_1, result_2)
            print(f"Result: Total ({total}), Executed ({executed}), Skipped ({skipped}), Original ({success_1}), New ({success_2}), Fixed ({fixed}), Broken ({broken})")
