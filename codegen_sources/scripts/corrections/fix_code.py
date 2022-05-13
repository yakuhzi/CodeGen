import re
from tree_sitter import Language, Parser, TreeCursor


TOFILL = {"python": "#TOFILL", "java": "//TOFILL", "cpp": "//TOFILL"}


def fix_code(script_model, f_fill, lang, lang_processor, f_name=None):
    #print("script_model", script_model)

    f_fill = lang_processor.detokenize_code(f_fill)
    f_fill = f_fill.replace(f_name, "f_filled")
    print("ORIGINAL", f_fill)

    code = bytes(f_fill, "utf8")

    parser = Parser()
    parser.set_language(Language('codegen_sources/scripts/parsing/build/library.so', lang))

    tree = parser.parse(code)
    cursor = tree.walk()

    for node in traverse_tree(cursor):
        snippet = code[node.start_byte: node.end_byte].decode("utf8")
        #print(node, snippet)

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
        if node.type == "call" and snippet.startswith("math.sqrt"):
            f_fill = f_fill.replace(snippet, f"int({snippet})")

        # Fix intersection instead of in
        if node.type == "call" and "intersection" in snippet:
            fixed_snippet = re.sub("(\w*)\.intersection \((.*?)\)", r"\2 in \1", snippet)
            print(node, snippet, fixed_snippet)
            f_fill = f_fill.replace(snippet, fixed_snippet)

    # Fix rounding issues
    f_fill = f_fill.replace("//", "/")

    # Fix tuple assignments
    f_fill = re.sub("^(\s*)(.*?),\s*(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\s*=\s*(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\s*$", r"\1\2 = \6\n\1\3 = \7\n\1\4 = \8\n\1\5 = \9", f_fill, flags=re.MULTILINE)
    f_fill = re.sub("^(\s*)(.*?),\s*(.*?)\s*,\s*(.*?)\s*=\s*(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\s*$", r"\1\2 = \5\n\1\3 = \6\n\1\4 = \7", f_fill, flags=re.MULTILINE)
    f_fill = re.sub("^(\s*)(.*?),\s*(.*?)\s*=\s*(.*?)\s*,\s*(.*?)\s*$", r"\1\2 = \4\n\1\3 = \5", f_fill, flags=re.MULTILINE)

    # Fix wrong initial value (min / max)
    f_fill = re.sub("^(\s*min.*=.*)-sys\.maxsize(.*)$", r"\1sys.maxsize\2", f_fill, flags=re.MULTILINE)
    f_fill = re.sub("^(\s*max.*=.*)[^-]sys\.maxsize(.*)$", r"\1-sys.maxsize\2", f_fill, flags=re.MULTILINE)

    # Fix global variable misuse
    f_fill = re.sub("^(\s*)global\s*(\w*)$", r"\1\2 = 0", f_fill, flags=re.MULTILINE)

    print("FIXED", f_fill)

    ret = script_model.replace(
        TOFILL[lang],
        "\n".join([f_fill, "\n"])
    )

    #print("ret", ret)
    return ret


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
