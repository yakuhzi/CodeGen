from . import parser
import re
from fuzzywuzzy import fuzz, process
from codegen_sources.model.src.utils import get_errors


def get_best_match(original_code, best_original_matches, best_generic_matches):
    match_scores = {}

    for match in best_original_matches:
        match_scores[match[2]] = match[1]

    for match in best_generic_matches:
        match_scores[match[2]] += match[1]

    best_index = max(match_scores, key=match_scores.get)
    best_match = original_code[best_index]
    print("BEST MATCH", best_match)
    return best_match


def correct_code():
    original_code = """public static int foo() {
    System.out.println(\"Hello World!\");
    int x = 23 * 5;
    int y = 2 / 4;
    return x * y;
}"""

    translated_code = """int foo() {
    std::cout << \"Hello World!\";
    int x = 23 * 5;
    int y = 2 / 
    return x * y;
}"""

    original_dict = {}
    translated_dict = {}

    for i, line in enumerate(original_code.split("\n")):
        original_dict[i] = line.strip() + "\n"

    for i, line in enumerate(translated_code.split("\n")):
        translated_dict[i] = line.strip() + "\n"

    compilation_errors, _ = get_errors(translated_code, "cpp")
    match = re.search(r"\.cpp:(\d+):\d+: error: ", compilation_errors)

    if not match:
        return

    affected_line = int(match.group(1)) - 10
    broken_translation = translated_dict[affected_line].strip()
    print("BROKEN", broken_translation)

    rule_set = parser.RuleSet()
    generics = {}

    for i, line in original_dict.items():
        generic = rule_set.create_generic_expression(line, parser.JAVA.lower())
        generics[i] = generic

    generic = rule_set.create_generic_expression(broken_translation, parser.CPP.lower())

    best_original_matches = process.extractBests(broken_translation, original_dict, scorer=fuzz.ratio)
    best_generic_matches = process.extractBests(generic, generics, scorer=fuzz.ratio)

    original_line = get_best_match(original_dict, best_original_matches, best_generic_matches)
    translation = rule_set.translate_line(original_line, parser.JAVA)
    
    print(translated_dict)
    translated_dict[affected_line] = translation[0][0]
    print(translated_dict)
    corrected_translation = ''.join(str(x) for x in translated_dict.values())
    print(corrected_translation)


if __name__ == "__main__":
    correct_code()