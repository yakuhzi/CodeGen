MODEL_PATH='models/'
INPUT_FILE=''

python -m codegen_sources.model.deobfuscate --lang python  --model_path $MODEL_PATH --beam_size 1 < $INPUT_FILE