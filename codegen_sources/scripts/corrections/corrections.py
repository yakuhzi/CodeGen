import os
from pathlib import Path

import torch

from ...model.src.evaluation.comp_acc_computation import run_python_program
from ...model.translate import Translator
from ...preprocessing.lang_processors.lang_processor import LangProcessor
from .fix_code import fix_code

TREE_SITTER_ROOT = Path(__file__).parents[3].joinpath("tree-sitter")
SRC_LANGUAGE = 'cpp'
TGT_LANGUAGE = 'python'
PATH = 'dataset/transcoder/test/transcoder_test.' + SRC_LANGUAGE + '.tok'

print(TREE_SITTER_ROOT)
src_lang_processor = LangProcessor.processors[SRC_LANGUAGE](root_folder=TREE_SITTER_ROOT)

with torch.no_grad():
    translator = Translator('models/transcoder_st/Online_ST_CPP_Python.pth', 'data/bpe/cpp-java-python/codes')

    with open(PATH) as file:
        total = 0
        skipped = 0
        executed = 0
        success_1 = 0
        success_2 = 0
        fixed = 0
        breaked = 0

        for line in file:
            function_id = line.split(" | ")[0]
            file_suffix = "py" if TGT_LANGUAGE == "python" else TGT_LANGUAGE
            original_eval_path = "data/transcoder_evaluation_gfg/" + TGT_LANGUAGE + "/" + function_id + "." + file_suffix
            filled_eval_path = "dump/transcoder_st/eval/cpp_python/online_st/20668773/eval_scripts/" + SRC_LANGUAGE + "_sa-" + TGT_LANGUAGE + "_sa.test/" + function_id + "." + file_suffix

            total += 1

            # whitelist = [
            #     "MAXIMUM_PRODUCT_SUBSET_ARRAY"
            # ]

            # if function_id not in whitelist:
            #     continue

            if not os.path.exists(filled_eval_path):
                skipped += 1
                continue

            print(f"Executing '{function_id}'")
            executed += 1

            function = " | ".join(line.split(" | ")[1:])
            function = src_lang_processor.detokenize_code(function)

            filled_script_model = open(filled_eval_path, "r").read()
            original_script_model = open(original_eval_path, "r").read()

            if TGT_LANGUAGE == "python":
                original_script_model = f"import numpy as np \nimport math\nfrom math import *\nimport collections\nfrom collections import *\nimport heapq\nimport itertools\nimport random\nimport sys\n\n{original_script_model}"

            output = translator.translate(
                function,
                lang1=SRC_LANGUAGE,
                lang2=TGT_LANGUAGE,
                beam_size=1,
            )

            f_fill = output[0]

            tgt_lang_processor = LangProcessor.processors[TGT_LANGUAGE](root_folder=TREE_SITTER_ROOT)
            f_name = tgt_lang_processor.get_function_name(f_fill)

            fixed_script_model = fix_code(original_script_model, f_fill, TGT_LANGUAGE, tgt_lang_processor, f_name=f_name)

            filled_script_path = "filled_script_path"
            fixed_script_path = "fixed_script_path"

            open(filled_script_path, "w", encoding="utf-8").write(filled_script_model)
            open(fixed_script_path, "w", encoding="utf-8").write(fixed_script_model)

            # print(filled_script_model)
            # print(fixed_script_model)

            result_1, _ = run_python_program(filled_script_path, 0)
            result_2, _ = run_python_program(fixed_script_path, 0)

            if result_2[0] != "success":
                pass
                # print(filled_script_model)
                # print(fixed_script_model)

            total += 1

            if result_1[0] == "success":
                success_1 += 1
            if result_2[0] == "success":
                success_2 += 1

            if result_1[0] != "success" and result_2[0] == "success":
                fixed += 1
            elif result_1[0] == "success" and result_2[0] != "success":
                breaked += 1

            print("Result: ", result_1, result_2)
            print(f"Result: Total ({total}), Executed ({executed}), Skipped ({skipped}), Original ({success_1}), New ({success_2}), Fixed ({fixed}), Breaked ({breaked})")

        print(f"Result: Total ({total}), Executed ({executed}), Skipped ({skipped}), Original ({success_1}), New ({success_2}), Fixed ({fixed}), Breaked ({breaked})")
