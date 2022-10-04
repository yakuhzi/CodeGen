import re
from typing import Tuple

from tree_sitter import Language, Parser, TreeCursor
from logging import getLogger

from ...preprocessing.lang_processors.lang_processor import LangProcessor

TOFILL = {"python": "#TOFILL", "java": "//TOFILL", "cpp": "//TOFILL"}

logger = getLogger()


def fix_code(f_fill: str, lang: str, errors: str) -> str:
    logger.debug("=" * 100)
    logger.debug(f"ORIGINAL\n{f_fill}")
    logger.debug("=" * 100)

    # Convert function string to bytes
    code = bytes(f_fill, "utf8")

    # Define AST parser
    parser = Parser()
    parser.set_language(Language('codegen_sources/scripts/corrections/build/library.so', lang))

    # Obtain AST tree and cursor
    tree = parser.parse(code)
    cursor = tree.walk()

    # Fix code depending on the target language
    if lang == "cpp":
        f_fill = fix_cpp_code(f_fill, cursor, code, errors)
    elif lang == "java":
        f_fill = fix_java_code(f_fill, cursor, code, errors)
    elif lang == "python":
        f_fill = fix_python_code(f_fill, cursor, code, errors)

    logger.debug("=" * 100)
    logger.debug(f"FIXED\n{f_fill}")
    logger.debug("=" * 100)
    return f_fill


def fix_cpp_code(f_fill: str, cursor: TreeCursor, code: bytes, errors: Tuple[str, str]) -> str:
    compile_errors, _ = errors
    logger.debug(f"COMPILE ERRORS: {compile_errors}")

    defined_variables = []

    for error in [x.group() for x in re.finditer("error: .*\n.*", compile_errors)]:
        match = re.search("error: ‘(\w+)’ was not declared in this scope", error)

        # Fix use of undefined variables
        if match and match.group(1) not in defined_variables:
            f_fill = re.sub("(\w+ f_filled \(.*\) {\n)", r"\1  " + f"int {match.group(1)} = 0 ;\n", f_fill)
            defined_variables.append(match.group(1))

        # Fix extra '}' before return
        if error.startswith("error: expected declaration before ‘}’ token"):
            f_fill = re.sub("}\n(return.*\n})", r"\1", f_fill)

        # Fix extra '('
        if error.startswith("error: expected ‘)’ before"):
            snippet = error.split("\n")[1].strip()
            fixed_snippet = re.sub("\( \( ", "( ", snippet, 1)
            f_fill = f_fill.replace(snippet, fixed_snippet)

    # Fix using Java types
    if "‘Boolean’ does not name a type" in compile_errors:
        f_fill = f_fill.replace("Boolean", "bool")

    if "‘Integer’ does not name a type" in compile_errors:
        f_fill = f_fill.replace("Integer", "int")

    if "‘String’ does not name a type" in compile_errors:
        f_fill = f_fill.replace("String", "string")

    # Fix wrong initial value (min / max)
    f_fill = re.sub("^(\s*int min.*=\s*[^-])INT_MIN", r"\1INT_MAX", f_fill, flags=re.MULTILINE)
    f_fill = re.sub("^(\s*int max.*=\s*[^-])INT_MAX", r"\1INT_MIN", f_fill, flags=re.MULTILINE)

    # Fix missing memset
    if "memset" not in f_fill:
        f_fill = re.sub(
            "(^( *)((int|long|double) )+(\w+) (\[ [\w\-+ ]* \] )+;)",
            r"\1\n\2memset ( \5, 0, sizeof ( \5 ) );",
            f_fill,
            flags=re.MULTILINE
        )

    # Fix multiple return statements
    f_fill = re.sub("^( *return.*)(\n* *return.*)+", r"\1", f_fill, flags=re.MULTILINE)

    while f_fill.count("{") > f_fill.count("}"):
        f_fill += "\n}"

    for node in traverse_tree(cursor):
        snippet = code[node.start_byte: node.end_byte].decode("utf8")

        # Fix additional boolean condition
        match = re.match("(\( (.*) ((>|<|=)+) (\+) \)) && \( (.*) ((>|<|=)+) (\w+) \)", snippet)

        if node.type == "binary_expression" and match:
            if match.group(2) == match.group(6) and match.group(5) == match.group(9):
                first_operator = match.group(3)
                second_operator = match.group(7)

                if first_operator == "<" and second_operator == ">=" \
                    or first_operator == "<=" and second_operator in [">", ">="] \
                    or first_operator == ">" and second_operator  == "<=" \
                    or first_operator == ">=" and second_operator in ["<", "<="]:
                    f_fill = f_fill.replace(snippet, match.group(1))

    return f_fill


def fix_java_code(f_fill: str, cursor: TreeCursor, code: bytes, errors: Tuple[str, str]) -> str:
    compile_errors, linting_errors = errors
    logger.debug(f"COMPILE ERRORS: {compile_errors}")

    # Fix type conversion errors
    for error in "\n".join([line.strip() for line in compile_errors.split("\n")]).split("^\n")[:-1]:
        error_code = error.split("\n")[1].strip().replace(";", "")
        match = re.search(
            r"from ((int)|(long)|(float)|(double)|(char)) to ((int)|(long)|(float)|(double)|(char))",
            error
        )

        if match:
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

        # Fix wrong log usage
        if "cannot find symbol" in error:
            f_fill = f_fill.replace("log10", "Math . log10")
            f_fill = f_fill.replace("log2", "log")

        f_fill = f_fill.replace(", Collections . reverseOrder ( )", "")

    # Fix reference function has different argument types
    # match = re.search("incompatible types: ((\+)[ [\]]*) cannot be converted to ((\w+)[ [\]]*)", errors)

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

    # Fix multiple return statements
    f_fill = re.sub("^( *return.*)(\n* *return.*)+", r"\1", f_fill, flags=re.MULTILINE)

    # Fix compare boolean with int
    f_fill = re.sub("((while|if) \(.*(<|>|<=|>=|==).*\)) (==|!=) \d* \)", r"\1 )", f_fill)

    if "int i " in f_fill:
        f_fill = re.sub("i == true", "i == 1", f_fill)
        f_fill = re.sub("i == false", "i == 0", f_fill)

    for node in traverse_tree(cursor):
        snippet = code[node.start_byte: node.end_byte].decode("utf8")

        # Fix Arrays.fill runtime errors
        if node.type == "expression_statement" and snippet.startswith("Arrays . fill "):
            array_vars = re.findall(r"\w+ (\w+) \[ \] \[ \] = new \w+ \[.*\] \[.*\] ;", f_fill, flags=re.MULTILINE)
            array_vars += re.findall(r"\w+ \[ \] \[ \] (\w+) = new \w+ \[.*\] \[.*\] ;", f_fill, flags=re.MULTILINE)
            match = re.search(r"Arrays \. fill \( (\w+) , \w+ \) ;", snippet)

            if match:
                array_var = match.group(1)

                if array_var in array_vars:
                    f_fill = f_fill.replace(snippet + "\n", "")

        # Fix unary operator used with bitwise operation as boolean expression
        if "for unary operator '!'" in compile_errors and node.type == "unary_expression":
            fixed_snippet = "( " + snippet.replace("!", "") + " ) == 0"
            f_fill = f_fill.replace(snippet, fixed_snippet)

        # Fix bitwise AND used in if condition
        if "if" in compile_errors and "&" in snippet and "&&" not in snippet and node.type == "parenthesized_expression":
            fixed_snippet = snippet.replace("(", "((").replace(")", ") == 1)")
            f_fill = f_fill.replace(snippet, fixed_snippet)

        # Fix unitinitialized variable
        match = re.search(r"variable (\w+) might not have been initialized", compile_errors)

        if match:
            variable = match.group(1)

            if re.match(f"((int)|(double)|(float)|(long)) {variable}", snippet) and node.type == "local_variable_declaration":
                fixed_snippet = snippet.replace(";", "= 0 ;")
                f_fill = f_fill.replace(snippet, fixed_snippet)

    return f_fill


def fix_python_code(f_fill: str, cursor: TreeCursor, code: bytes, errors: Tuple[str, str]) -> str:
    compile_errors, linting_errors = errors
    logger.debug(f"COMPILE ERRORS: {compile_errors}")
    logger.debug(f"LINTING ERRORS: {linting_errors}")

    # Fix usage of variable before assignment
    for error in linting_errors.split("\n"):
        match = re.search("Using variable '(\w+)' before assignment", error)
        defined_variables = []

        if match and match.group(1) not in defined_variables:
            f_fill = re.sub("(def f_filled \( .* \) :\n)", r"\1    " + f"{match.group(1)} = 0\n", f_fill)
            defined_variables.append(match.group(1))

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
                    fixed_snippet = left_side + "=" + right_side + " , 0"
                    f_fill = f_fill.replace(snippet, fixed_snippet)

        # Fix intersection instead of in
        if node.type == "call" and "intersection" in snippet:
            fixed_snippet = re.sub("(\w+)\.intersection \((.*?)\)", r"\2 in \1", snippet)
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
    f_fill = re.sub("^(\s*)global\s*(\w+)$", r"\1\2 = 0", f_fill, flags=re.MULTILINE)

    # Fix multiple return statements
    f_fill = re.sub("^(    return.*)(\n*    return.*)+", r"\1", f_fill, flags=re.MULTILINE)

    # Fix swap
    f_fill = re.sub("swap \( (\w+) , (\w+) \)", r"\1, \2 = \2, \1", f_fill)

    # Fix missing typecast to int for math.sqrt in range of for loop
    f_fill = re.sub("(for .* range \(.*)(math.sqrt \( .*? \))(.*\) :)", r"\1int ( \2 )\3", f_fill)
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
