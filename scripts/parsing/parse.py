import os
import torch
from tree_sitter import Language, Parser, TreeCursor

from codegen_sources.model.translate import Translator

DATASET_PATH = 'dataset/transcoder/test'
MODEL_PATH='models/transcoder/TransCoder_model_1.pth'
BPE_PATH = 'data/bpe/cpp-java-python/codes'

def translate(code: str, src_lang: str, tgt_lang: str, beam_size: int = 1):
    translator = Translator(MODEL_PATH, BPE_PATH)

    print(f"Input {src_lang} function:")
    print(code)
    
    with torch.no_grad():
        output = translator.translate(
            input,
            lang1=src_lang,
            lang2=tgt_lang,
            beam_size=beam_size,
        )

    print(f"Translated {tgt_lang} function:")

    for out in output:
        print("=" * 20)
        print(out)
        
def parse_cpp(code: str, cursor: TreeCursor):
    print("========================================")
    assert cursor.goto_first_child()
    print(code[cursor.node.start_byte: cursor.node.end_byte])


def parse_java(code: str, cursor: TreeCursor):
    print("========================================")
    
    assert cursor.goto_first_child()
    print(code[cursor.node.start_byte: cursor.node.end_byte])
    assert cursor.goto_next_sibling()
    print(code[cursor.node.start_byte: cursor.node.end_byte])
    assert cursor.goto_next_sibling()
    print(code[cursor.node.start_byte: cursor.node.end_byte])
    assert cursor.goto_first_child()
    print(code[cursor.node.start_byte: cursor.node.end_byte])

    while cursor.goto_next_sibling():
        print(code[cursor.node.start_byte: cursor.node.end_byte])


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
                return
            elif language == "java":
                parse_java(code, cursor)
            elif language == "python":
                parse_python(code, cursor)

if __name__ == "__main__":
    Language.build_library(
        'scripts/parsing/build/library.so',
        [
            'tree-sitter/tree-sitter-cpp',
            'tree-sitter/tree-sitter-java',
            'tree-sitter/tree-sitter-python'
        ]
    )

    for subdir, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if not file.endswith("java.tok"):
                continue

            parse_file(os.path.join(subdir, file))
