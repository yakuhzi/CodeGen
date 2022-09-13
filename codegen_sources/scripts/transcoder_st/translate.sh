CPP_INPUT_FILE='codegen_sources/scripts/test_input/test.cpp'
JAVA_INPUT_FILE='codegen_sources/scripts/test_input/Test.java'
PYTHON_INPUT_FILE='codegen_sources/scripts/test_input/test.py'

# ===============
# CPP -> Java
# ===============
# MODEL_PATH='models/Online_ST_CPP_Java.pth'
# python -m codegen_sources.model.translate --src_lang cpp --tgt_lang java --model_path "$MODEL_PATH" --beam_size 1 < "$CPP_INPUT_FILE"

# ===============
# CPP -> Python
# ===============
# MODEL_PATH='models/Online_ST_CPP_Python.pth'
# python -m codegen_sources.model.translate --src_lang cpp --tgt_lang python --model_path "$MODEL_PATH" --beam_size 1 < "$CPP_INPUT_FILE"

# ===============
# Java -> CPP
# ===============
# MODEL_PATH='models/Online_ST_Java_CPP.pth'
# python -m codegen_sources.model.translate --src_lang java --tgt_lang cpp --model_path "$MODEL_PATH" --beam_size 1 < "$JAVA_INPUT_FILE"

# ===============
# Java -> Python
# ===============
# MODEL_PATH='models/Online_ST_Java_Python.pth'
# python -m codegen_sources.model.translate --src_lang java --tgt_lang python --model_path "$MODEL_PATH" --beam_size 1 < "$JAVA_INPUT_FILE"

# ===============
# Python -> CPP
# ===============
# MODEL_PATH='models/Online_ST_Python_CPP.pth'
# python -m codegen_sources.model.translate --src_lang python --tgt_lang cpp --model_path "$MODEL_PATH" --beam_size 1 < "$PYTHON_INPUT_FILE"

# ===============
# Python -> Java
# ===============
# MODEL_PATH='models/Online_ST_Python_Java.pth'
# python -m codegen_sources.model.translate --src_lang python --tgt_lang java --model_path "$MODEL_PATH" --beam_size 1 < "$PYTHON_INPUT_FILE"