# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import subprocess

import json
from concurrent.futures import ProcessPoolExecutor
import sys

import os

from codegen_sources.model.src.utils import get_errors, has_compile_errors
from ....scripts.corrections.fix_code import fix_code
from logging import getLogger
from tqdm import tqdm

from ..utils import (
    REPO_ROOT,
    limit_virtual_memory,
    MAX_VIRTUAL_MEMORY,
    read_file_lines,
    get_java_bin_path,
    has_compile_errors,
)

sys.path.append(str(REPO_ROOT))
print("adding to path", str(REPO_ROOT))
TREE_SITTER_ROOT = REPO_ROOT.joinpath("tree-sitter")
import codegen_sources.preprocessing.lang_processors.cpp_processor
import codegen_sources.preprocessing.lang_processors.java_processor
import codegen_sources.preprocessing.lang_processors.python_processor
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor

from codegen_sources.test_generation.test_runners.cpp_test_runner import CppTestRunner
from codegen_sources.test_generation.test_runners.python_test_runner import (
    PythonTestRunner,
)
from codegen_sources.test_generation.evosuite_tests_translators.evosuite_to_python import (
    EvosuiteToPython,
)
from codegen_sources.test_generation.evosuite_tests_translators.evosuite_to_cpp import (
    EvosuiteToCpp,
)

EXT = {"python": "py", "java": "java", "cpp": "cpp"}

TOFILL = {"python": "#TOFILL", "java": "//TOFILL", "cpp": "//TOFILL"}

primitive_types = {"short", "int", "long", "float", "double", "boolean", "char"}

EVOSUITE_TESTS_TRANSCODER_PATH = (
    REPO_ROOT.joinpath("data")
    .joinpath("evosuite_unit_tests")
    .joinpath("transcoder_test_set.json")
)

logger = getLogger()


def eval_state(proc, proc_name):
    try:
        try:
            result, stderr = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            c = (
                "kill `ps aux | grep '"
                + proc_name
                + "' | grep -v jupyter | grep -v grep | awk '{print($2)}'`"
            )
            subprocess.run(
                c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return "timeout", None
        results = result.decode("utf8", errors="replace")
        success, n_test = results.split("#Results:")[-1].split(",")
        if int(success) == int(n_test):
            return "success", None
        else:
            return "failure", result.decode("utf-8", errors="replace")
    except KeyboardInterrupt:
        raise
    except:
        return "error", stderr.decode("utf-8", errors="replace")


def run_python_program(script_path, i):
    proc = subprocess.Popen(
        f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY)}; python {script_path}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
    )
    res = eval_state(proc, f"python {script_path}")
    return res, i

def run_java_program(script_path, i, javafx_path: str):
    folder = os.path.dirname(script_path)
    name = os.path.basename(script_path).split(".")[0]

    proc = subprocess.Popen(
        f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY)}; cd {folder} &&  {os.path.join(get_java_bin_path(), 'javac')} --module-path {javafx_path} --add-modules javafx.base {name}.java && {os.path.join(get_java_bin_path(), 'java')} --module-path {javafx_path} --add-modules javafx.base {name}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
    )
    res = eval_state(proc, f"java {name}")
    return res, i


def run_cpp_program(script_path, i):
    folder = os.path.dirname(script_path)
    name = os.path.basename(script_path).split(".")[0]
    proc = subprocess.Popen(
        f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY)}; cd {folder} && g++ {name}.cpp -o {name}_cpp && ./{name}_cpp",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
    )
    res = eval_state(proc, f"{name}_cpp")
    return res, i


def make_arg_string(argtype, argval):
    if "[" not in argtype:
        return f"{argtype} {argval}"

    dim = argtype.count("[")
    argtype = argtype.replace("[", "").replace("]", "")
    return f'{argtype} {argval} {"[ ]" * dim}'


def submit_evosuite_functions(
    functions_list, id, lang, test_dictionary, roberta_mode=False
):
    assert lang in {"cpp", "python"}, f"{lang} is not supported for evosuite tests"
    if lang == "cpp":
        test_runner = CppTestRunner(timeout=30, compilation_timeout=30)
    else:
        assert lang == "python"
        test_runner = PythonTestRunner(timeout=30)
    lang_processor = LangProcessor.processors[lang](root_folder=TREE_SITTER_ROOT)
    id = id.rstrip()
    if id not in test_dictionary or test_dictionary[id] == "missing":
        return [return_script_not_found()], id
    test = test_dictionary[id]
    results_list = []
    for try_id, f_fill in enumerate(functions_list):
        f = f_fill.rstrip()
        f = (
            lang_processor.detokenize_code(f)
            if not roberta_mode
            else f.replace("#NEWLINE", "\n")
        )
        result = test_runner.get_tests_results(f, test)
        results_list.append((result[0], None))
        if result[0] == "success":
            return results_list, id
    return results_list, id


def submit_functions(
    functions_list,
    id,
    ref,
    lang1,
    lang2,
    outfolder,
    script_folder,
    roberta_mode=False,
    correct_functions=False,
    replaced_constrained=False,
    replaced_knnmt=False,
):
    lang_processor = LangProcessor.processors[lang2](root_folder=TREE_SITTER_ROOT)
    results_list = []
    i = id.rstrip()

    for try_id, f_fill in enumerate(functions_list):
        f_fill = f_fill.rstrip()
        ref = ref.rstrip()

        script_model_path = os.path.join(script_folder, f"{lang2}/{i}.{EXT[lang2]}")

        if not os.path.exists(script_model_path):
            return [return_script_not_found()], i, 0, False, False

        script_model = open(script_model_path, "r", encoding="utf-8").read()

        try:
            f_name = lang_processor.get_function_name(f_fill)
            f_fill = f_fill.replace(f_name, "f_filled")
        except:
            results_list.append(("error", "Could not replace function name"))
            f_name = ""

        f_fill = (lang_processor.detokenize_code(f_fill) if not roberta_mode else f_fill.replace("#NEWLINE", "\n"))
        ref = (lang_processor.detokenize_code(ref) if not roberta_mode else ref.replace("#NEWLINE", "\n"))

        run_pg = globals()[f"run_{lang2}_program"]

        if lang2 == "python":
            script_model = f"import numpy as np \nimport math\nfrom math import *\nimport collections\nfrom collections import *\nimport heapq\nimport itertools\nimport random\nimport sys\n\n{script_model}"

        script_path = f"{outfolder}/{i}.{EXT[lang2]}"
        javafx_path = os.path.abspath("javafx-sdk-11/lib")

        if correct_functions:
            # Get original results without the rule-based corrections applied
            original_f = f_fill
            script = script_model.replace(TOFILL[lang2], original_f)
            open(script_path, "w", encoding="utf-8").write(script)

            if lang2 == "java":
                result, _ = run_pg(script_path, i, javafx_path)
            else:
                result, _ = run_pg(script_path, i)

            original_success = result[0] == "success"
            os.remove(script_path)

            # Get corrected function
            errors = get_errors(f_fill, tgt_language=lang2)
            f_fill = fix_code(f_fill, lang2, errors)

        if f_fill == ref:
            results_list.append(("success", "identical to gold"))

            if correct_functions and not original_success:
                logger.debug(f"Fixed function through rule based fixes ({try_id}, {i}):\n{original_f}\n{f_fill}\n{ref}")
                return results_list, i, 1, False, False

            if replaced_constrained:
                logger.debug(f"Fixed function through constrained beam search ({try_id}, {i}):\n{f_fill}\n{ref}")
                return results_list, i, 0, True, False

            if replaced_knnmt:
                logger.debug(f"Fixed function through KNNMT ({try_id}, {i}):\n{f_fill}\n{ref}")
                return results_list, i, 0, False, True

            return results_list, i, 0, False, False

        script = script_model.replace(TOFILL[lang2], f_fill)
        open(script_path, "w", encoding="utf-8").write(script)

        if lang2 == "java":
            result, _ = run_pg(script_path, i, javafx_path)
        else:
            result, _ = run_pg(script_path, i)

        if result[0] == "success":
            results_list.append(result)

            if correct_functions and not original_success:
                logger.debug(f"Fixed function through rule based fixes ({try_id}, {i}):\n{original_f}\n{f_fill}\n{ref}")
                return results_list, i, 1, False, False

            if replaced_constrained:
                logger.debug(f"Fixed function through constrained beam search ({try_id}, {i}):\n{f_fill}\n{ref}")
                return results_list, i, 0, True, False

            if replaced_knnmt:
                logger.debug(f"Fixed function through KNNMT ({try_id}, {i}):\n{f_fill}\n{ref}")
                return results_list, i, 0, False, True

            return results_list, i, 0, False, False

        if has_compile_errors(f_fill, tgt_language=lang2):
            result = list(result)
            result[0] = "compile_error"
            result = tuple(result)

            # unsuccessful_path = f"unsuccessful.{lang1}_{lang2}"
            # file = open(unsuccessful_path, "a")
            # file.write(i + "\n")
            # file.close()

        results_list.append(result)

        if correct_functions and original_success:
            logger.debug(f"Corrupted function with rule based fixes ({try_id}, {i}):\n{original_f}\n{f_fill}\n{ref}")
            return results_list, i, -1, False, False

    return results_list, i, 0, False, False


def eval_function_output(
    ref_path,
    hyp_paths,
    id_path,
    lang1,
    lang2,
    outfolder,
    script_folder,
    roberta_mode,
    evosuite_functions=False,
    evosuite_tests=None,
    correct_functions=False,
    constrained=False,
    knnmt_paths=None,
    knnmt_restricted=True
):
    # List of tuples of functions. List elements = function id, tuple elements = beam.
    functions = list(zip(*[read_file_lines(path) for path in hyp_paths]))
    # List of function ids
    ids = read_file_lines(id_path)
    # List of reference functions
    refs = read_file_lines(ref_path)

    assert len(functions) == len(ids), f"{len(functions), len(ids)}"
    assert len(functions) == len(refs), f"{len(functions), len(refs)}"

    if knnmt_paths is not None:
        # List of tuples of KNNMT functions. List elements = function id, tuple elements = beam.
        knnmt_functions = list(zip(*[read_file_lines(path) for path in knnmt_paths]))

        assert len(knnmt_functions) == len(ids), f"{len(functions), len(ids)}"
        assert len(knnmt_functions) == len(refs), f"{len(functions), len(refs)}"
    else:
        knnmt_functions = [None for f in functions]

    lang1 = lang1.split("_")[0]
    lang2 = lang2.split("_")[0]
    jobs = []
    executor = ProcessPoolExecutor()

    # For each function in list
    for f, knnmt_f, i, r in zip(functions, knnmt_functions, ids, refs):
        replaced_constrained = False
        replaced_knnmt = False
        f = list(f)

        already_compiles = not has_compile_errors(f[0], tgt_language=lang2)

        # Constrained Beam Search
        if constrained and not already_compiles:
            # For each beam in tuple
            for index in range(1, len(f)):
                # Use first function in the beam that compiles successfully
                if not has_compile_errors(f[index], tgt_language=lang2):
                    f = [f[index]]
                    replaced_constrained = True
                    break

        # Nearest Neighbor Machine Translation
        if knnmt_f is not None and (not knnmt_restricted or not already_compiles) and not replaced_constrained:
            # For each beam in tuple
            for knnmt_function in knnmt_f:
                # Replace beam with kNN-MT translation
                if not constrained or not has_compile_errors(knnmt_function, tgt_language=lang2):
                    f = [knnmt_function]
                    replaced_knnmt = True
                    break

        # Alternative: Instead searching all functions to find the first one that successfully compiles,
        # iteratively search both in the original translations and in the kNN-MT translations.
        # if (constrained or knnmt_f is not None) and has_compile_errors(f[0], tgt_language=lang2):
        #     # For each beam in tuple
        #     for index, function in enumerate(f):
        #         if constrained and not (has_compile_errors(function, tgt_language=lang2) if index > 0 else True):
        #             f = [function]
        #             replaced_constrained = True
        #             break

        #         if knnmt_f is not None and (not constrained or not has_compile_errors(knnmt_f[index], tgt_language=lang2)):
        #             f = [knnmt_f[index]]
        #             replaced_knnmt = True
        #             break

        # Use first function in the beam if not replaced by constrained beam search or kNN-MT
        if not replaced_constrained and not replaced_knnmt:
            f = [f[0]]

        f = tuple(f)

        if evosuite_functions:
            jobs.append(
                executor.submit(
                    submit_evosuite_functions,
                    f,
                    i,
                    lang2,
                    evosuite_tests[lang2],
                    roberta_mode
                )
            )
        else:
            jobs.append(
                executor.submit(
                    submit_functions,
                    f,
                    i,
                    r,
                    lang1,
                    lang2,
                    outfolder,
                    script_folder,
                    roberta_mode,
                    correct_functions,
                    replaced_constrained,
                    replaced_knnmt
                )
            )

    results_stats = {
        "success": 0,
        "failure": 0,
        "error": 0,
        "timeout": 0,
        "script_not_found": 0,
        "identical_gold": 0,
        "compile_error": 0,
        "fixed_rules": 0,
        "corrupted_rules": 0,
        "fixed_constrained": 0,
        "fixed_knnmt": 0
    }

    results = ["" for _ in range(len(ids))]
    for job in tqdm(jobs):
        results_list, i, rules_result, fixed_constrained, fixed_knnmt = job.result()
        nb_success = sum([r[0] == "success" for r in results_list])
        nb_identical = sum(
            [r[0] == "success" and r[1] == "identical to gold" for r in results_list]
        )
        assert nb_success <= 1, "Should stop after first success"
        if nb_success > 0:
            results_stats["success"] += 1
            if nb_identical > 0:
                results_stats["identical_gold"] += 1
        else:
            results_stats[results_list[0][0]] = (
                results_stats.get(results_list[0][0], 0) + 1
            )

        # Count statistics for examined improvements
        results_stats["fixed_rules"] += 1 if rules_result == 1 else 0
        results_stats["corrupted_rules"] += 1 if rules_result == -1 else 0
        results_stats["fixed_constrained"] += 1 if fixed_constrained else 0
        results_stats["fixed_knnmt"] += 1 if fixed_knnmt else 0

        results[ids.index(i + "\n")] = []
        for result, stderr in results_list:
            if stderr is not None:
                stderr = stderr.replace("\n", " ")
            else:
                stderr = "None"
            results[ids.index(i + "\n")].append(f"{result} : {stderr}")

    results_stats["total"] = len(functions)
    results_stats["total_evaluated"] = (
        len(functions) - results_stats["script_not_found"]
    )
    results_stats = {k: results_stats[k] for k in sorted(results_stats.keys())}
    return results_stats, results


def load_evosuite_transcoder_tests():
    cpp_test_translator = EvosuiteToCpp()
    python_test_translator = EvosuiteToPython()
    tests = {"java": {}, "java_scaffolding": {}, "python": {}, "cpp": {}}
    with open(EVOSUITE_TESTS_TRANSCODER_PATH, "r") as f:
        for l in f:
            json_line = json.loads(l)
            if json_line["tests_strings"] == "missing":
                continue
            tests["java"][json_line["TARGET_CLASS"]] = json_line["tests_strings"]
            tests["java_scaffolding"][json_line["TARGET_CLASS"]] = json_line[
                "scaffoldings_strings"
            ]
            python_test = python_test_translator.translate(json_line["tests_strings"])
            if not python_test_filter(python_test):
                continue
            tests["python"][json_line["TARGET_CLASS"]] = python_test

            cpp_test = cpp_test_translator.translate(json_line["tests_strings"])
            tests["cpp"][json_line["TARGET_CLASS"]] = cpp_test
    return tests


def python_test_filter(python_test):
    return (
        python_test.count("try ") == 0
        and python_test.count("catch(") == 0
        and python_test.count("assert ") > 0
    )


def return_script_not_found():
    return "script_not_found", None


def get_return_type(tokenized_java):
    return tokenized_java.split("(")[0].split()[-2]
