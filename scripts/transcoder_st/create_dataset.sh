JAVA_FUNC_DATASET='dataset/'
MODELS_PATH='models/transcoder_st'
OUTPUT_DIRECTORY='output/transcoder_st'

# Create data (it will take a while)
bash codegen_sources/test_generation/create_self_training_dataset.sh $JAVA_FUNC_DATASET $MODELS_PATH $OUTPUT_DIRECTORY