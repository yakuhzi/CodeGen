import re
from compileall import compile_file
from typing import Tuple

from tree_sitter import Language, Node, Parser, TreeCursor

from ...preprocessing.lang_processors.lang_processor import LangProcessor

TOFILL = {"python": "#TOFILL", "java": "//TOFILL", "cpp": "//TOFILL"}


def fix_code(
    script_model: str,
    f_fill: str,
    lang: str,
    lang_processor: LangProcessor,
    f_name: str,
    errors: str
) -> str:
    f_fill = lang_processor.detokenize_code(f_fill)
    f_fill = f_fill.replace(f_name, "f_filled")
    print("ORIGINAL\n", f_fill)
    print("=" * 40)

    code = bytes(f_fill, "utf8")

    parser = Parser()
    parser.set_language(Language('codegen_sources/scripts/parsing/build/library.so', lang))

    tree = parser.parse(code)
    cursor = tree.walk()

    if lang == "cpp":
        f_fill = fix_cpp_code(f_fill, cursor, code, errors)
    elif lang == "java":
        f_fill = fix_java_code(f_fill, cursor, code, errors)
    elif lang == "python":
        f_fill = fix_python_code(f_fill, cursor, code, errors)

    print("FIXED\n", f_fill)
    return script_model.replace(TOFILL[lang], "\n".join([f_fill, "\n"]))


def fix_cpp_code(f_fill: str, cursor: TreeCursor, code: bytes, errors: Tuple[str, str]) -> str:
    compile_errors, linting_errors = errors
    print("COMPILE ERRORS:", compile_errors)

    if "‘Boolean’ does not name a type" in compile_errors:
        f_fill = f_fill.replace("Boolean", "bool")

    if "Integer does not name a type" in compile_errors:
        f_fill = f_fill.replace("Integer", "int")

    if "String does not name a type" in compile_errors:
        f_fill = f_fill.replace("String", "string")

    return f_fill


def fix_java_code(f_fill: str, cursor: TreeCursor, code: bytes, errors: Tuple[str, str]) -> str:
    compile_errors, linting_errors = errors
    print("COMPILE ERRORS:", compile_errors)

    # Fix type conversion errors
    for error in "\n".join([line.strip() for line in compile_errors.split("\n")]).split("^\n")[:-1]:
        match = re.search(
            r"from ((int)|(long)|(float)|(double)|(char)) to ((int)|(long)|(float)|(double)|(char))",
            error
        )
        error_code = error.split("\n")[1].strip().replace(";", "")

        if match:
            used_type = match.group(1)
            expected_type = match.group(7)

            if "=" in error_code:
                left_side = error_code.split("=")[0]
                right_side = error_code.split("=")[1]
                fixed_code = f"{left_side} = ( {expected_type} ) ( {right_side} )"
                f_fill = f_fill.replace(error_code, fixed_code)
            elif "return" in error_code:
                right_side = error_code.split("return")[1]
                fixed_code = f"return ( {expected_type} ) ( {right_side} )"
                f_fill = f_fill.replace(error_code, fixed_code)

        # Fix ',' instead of ';' between two statements in one line
        if "error: ';' expected" in error:
            fixed_code = error_code.replace(",", ";")
            f_fill = f_fill.replace(error_code, fixed_code)

    # Fix reference function has different argument types
    # match = re.search("incompatible types: ((\w*)[ [\]]*) cannot be converted to ((\w*)[ [\]]*)", errors)

    # if match:
    #     f_fill = re.sub("(?<!(for \( )|(out\.pr))" + match.group(3).replace("[", "\\[").replace("]", "\\]"), match.group(1).lower(), f_fill)
    #     f_fill = re.sub("(?<!(for \( )|(out\.pr))" + match.group(4), match.group(2).lower(), f_fill)

    #     if match.group(2) == "int" and match.group(4) == "double":
    #         f_fill = f_fill.replace("Double .", "Integer .")
    #     elif match.group(2) == "double" and match.group(4) == "int":
    #         f_fill = f_fill.replace("Integer .", "Double .")
    #     elif match.group(1) == "char[]" and match.group(3) == "String":
    #         f_fill = re.sub(". charAt \( ([\w+\- ]*) \)", r"[ \1 ]", f_fill)

    # Fix wrong range
    # if result[0] == "failure":
    #     f_fill = re.sub("(for \(.*\s.*)(<=)", r"\1<", f_fill)

    for node in traverse_tree(cursor):
        snippet = code[node.start_byte: node.end_byte].decode("utf8")

        # Fix Arrays.fill runtime errors
        # if "Arrays.fill" in errors and node.type == "expression_statement" and snippet.startswith("Arrays . fill "):
        #     f_fill = f_fill.replace(snippet, "")

        # Fix unary operator used as boolean expression
        if "for unary operator '!'" in compile_errors and node.type == "unary_expression":
            fixed_snippet = "( " + snippet.replace("!", "") + " ) == 0"
            f_fill = f_fill.replace(snippet, fixed_snippet)

        # Fix unary used in if condition
        if "if" in compile_errors and "&" in snippet and "&&" not in snippet and node.type == "parenthesized_expression":
            fixed_snippet = snippet.replace("(", "((").replace(")", ") == 1)")
            f_fill = f_fill.replace(snippet, fixed_snippet)

        # Fix unitinitialized variable
        match = re.search(r"variable (\w*) might not have been initialized", compile_errors)

        if match:
            variable = match.group(1)

            if re.match(f"((int)|(double)|(float)|(long)) {variable}", snippet) and node.type == "local_variable_declaration":
                fixed_snippet = snippet.replace(";", "= 0 ;")
                f_fill = f_fill.replace(snippet, fixed_snippet)

    return f_fill


def fix_python_code(f_fill: str, cursor: TreeCursor, code: bytes, errors: Tuple[str, str]) -> str:
    compile_errors, linting_errors = errors
    print("COMPILE ERRORS:", compile_errors)
    print("LINTING ERRORS:", linting_errors)

    for node in traverse_tree(cursor):
        snippet = code[node.start_byte: node.end_byte].decode("utf8")

        # Fix tuple unpacking
        if node.type == "expression_statement" and "=" in snippet:
            snippet = code[cursor.node.start_byte: cursor.node.end_byte].decode("utf8")
            left_side = snippet.split("=")[0]
            right_side = snippet.split("=")[1]

            if "(" not in snippet:
                if left_side.count(',') < right_side.count(','):
                    fixed_snippet = left_side + "=" + ",".join(right_side.split(",")[:left_side.count(',') + 1])
                    f_fill = f_fill.replace(snippet, fixed_snippet)
                elif left_side.count(',') > right_side.count(','):
                    fixed_snippet = ",".join(left_side.split(",")[:right_side.count(',') + 1]) + "=" + right_side
                    f_fill = f_fill.replace(snippet, fixed_snippet)

        # Fix missing typecast to int for math.sqrt
        # if node.type == "call" and snippet.startswith("math.sqrt"):
        #     f_fill = f_fill.replace(snippet, f"int( {snippet} )")

        # Fix intersection instead of in
        if node.type == "call" and "intersection" in snippet:
            fixed_snippet = re.sub("(\w*)\.intersection \((.*?)\)", r"\2 in \1", snippet)
            f_fill = f_fill.replace(snippet, fixed_snippet)

    # Fix rounding issues
    # f_fill = f_fill.replace("//", "/")

    # Fix tuple assignments
    f_fill = re.sub(
        "^(\s*)(.*?),\s*(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\s*=\s*(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\s*$",
        r"\1\2 = \6\n\1\3 = \7\n\1\4 = \8\n\1\5 = \9",
        f_fill,
        flags=re.MULTILINE
    )
    f_fill = re.sub(
        "^(\s*)(.*?),\s*(.*?)\s*,\s*(.*?)\s*=\s*(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\s*$",
        r"\1\2 = \5\n\1\3 = \6\n\1\4 = \7",
        f_fill,
        flags=re.MULTILINE
    )
    f_fill = re.sub(
        "^(\s*)(.*?),\s*(.*?)\s*=\s*(.*?)\s*,\s*(.*?)\s*$",
        r"\1\2 = \4\n\1\3 = \5",
        f_fill,
        flags=re.MULTILINE
    )

    # Fix wrong initial value (min / max)
    f_fill = re.sub("^(\s*min.*=.*)-sys\.maxsize(.*)$", r"\1sys.maxsize\2", f_fill, flags=re.MULTILINE)
    f_fill = re.sub("^(\s*max.*=.*)[^-]sys\.maxsize(.*)$", r"\1-sys.maxsize\2", f_fill, flags=re.MULTILINE)

    # Fix global variable misuse
    f_fill = re.sub("^(\s*)global\s*(\w*)$", r"\1\2 = 0", f_fill, flags=re.MULTILINE)
    return f_fill


def traverse_tree(cursor: TreeCursor):
    reached_root = False
    while reached_root == False:
        yield cursor.node

        if cursor.goto_first_child():
            continue

        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True

            if cursor.goto_next_sibling():
                retracing = False
