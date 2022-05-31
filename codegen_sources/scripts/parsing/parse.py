import os
import sys
from unittest import result
import torch
from tree_sitter import Language, Parser, TreeCursor
from codegen_sources.model.translate import Translator


DATASET_PATH = 'dataset/transcoder/test'
MODEL_PATH = 'models/transcoder/TransCoder_model_1.pth'
BPE_PATH = 'data/bpe/cpp-java-python/codes'


def traverse_tree(cursor: TreeCursor, retracing=False):
    nodes = []

    if not retracing and cursor.node.type != "block" and cursor.goto_first_child():
        return traverse_tree(cursor)

    if cursor.node.type != "block" and cursor.goto_next_sibling():
        return traverse_tree(cursor)

    if cursor.node.type != "block" and cursor.goto_parent():
        return traverse_tree(cursor, True)

    
    cursor.goto_first_child()
    nodes.append(cursor.node)

    while cursor.goto_next_sibling():
        nodes.append(cursor.node)

    return (n for n in nodes if n.type != "{" and n.type != "}")

def parse_cpp(code: str, cursor: TreeCursor):
    print("========================================")
    assert cursor.goto_first_child()
    # print(code[cursor.node.start_byte: cursor.node.end_byte])


def parse_java(code: str, cursor: TreeCursor):
    print("========================================")
    input = []
    output = []

    for node in traverse_tree(cursor):
        snippet = code[node.start_byte: node.end_byte].decode("utf-8")
        input.append(snippet)

        with torch.no_grad():
            result = translator.translate(
                f"public static int a() {{\n    {snippet}\n}}",
                lang1='java',
                lang2='python',
                beam_size=1,
            )

            translation = '\n'.join(result[0].replace("\n\n", "\n").split("\n")[1:-1])

            if "return" in translation.split("\n")[-1] and snippet.count("return") != translation.count("return"):
                translation = '\n'.join(translation.split("\n")[:-1])

            output.append(translation)

    print("=" * 100)
    print("INPUT")
    print("\n".join(input))

    print("=" * 100)
    print("OUTPUT")
    print("\n".join(output))


def parse_python(code: str, cursor: TreeCursor):
    print("========================================")


def parse_file(path: str):
    language = path.split(".")[-2]
    parser = Parser()
    parser.set_language(Language('scripts/parsing/build/library.so', language))

    with open(path) as file:
        for line in file:
            function_id = line.split(" | ")[0]
            function = " | ".join(line.split(" | ")[1:])

            code = bytes(function, "utf8")
            tree = parser.parse(code)
            cursor = tree.walk()

            if language == "cpp":
                parse_cpp(code, cursor)
            elif language == "java":
                parse_java(code, cursor)
            elif language == "python":
                parse_python(code, cursor)
            return


if __name__ == "__main__":
    Language.build_library(
        'scripts/parsing/build/library.so',
        [
            'tree-sitter/tree-sitter-cpp',
            'tree-sitter/tree-sitter-java',
            'tree-sitter/tree-sitter-python'
        ]
    )

    # Initialize translator
    translator = Translator('models/transcoder_st/Online_ST_Java_Python.pth', 'data/bpe/cpp-java-python/codes')

    for subdir, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if not file.endswith("java.tok"):
                continue

            parse_file(os.path.join(subdir, file))
