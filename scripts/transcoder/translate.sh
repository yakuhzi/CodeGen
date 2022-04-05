MODEL_PATH='models/transcoder/TransCoder_model_1.pth'
INPUT_FILE='test.py'

python -m codegen_sources.model.translate --src_lang python --tgt_lang java --model_path $MODEL_PATH --beam_size 1 < $INPUT_FILE