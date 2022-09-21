# Conda environment
conda create --name cg python=3.6.9
conda activate cg
conda config --add channels conda-forge
conda config --add channels pytorch

# PyTorch with CUDA support
pip install torch==1.10.2+cu113 torchvision torchaudio -f https://download.pytorch.org/whl/cu113/torch_stable.html
conda install cudatoolkit=11.3 cudatoolkit-dev=11.3

# Other dependencies
conda install six scikit-learn stringcase ply slimit astunparse submitit
pip install transformers cython sacrebleu=="1.2.11" javalang tree_sitter psutil fastBPE
pip install hydra-core --upgrade --pre
pip install black==19.10b0 pylint pandas faiss-gpu tdqm pycuda pytorch_lightning torch_scatter thefuzz

# FastBPE
cd codegen_sources/model/tools
git clone https://github.com/glample/fastBPE.git
cd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
python setup.py install
cd ../../../../

# TreeSitter
mkdir tree-sitter
cd tree-sitter
git clone https://github.com/tree-sitter/tree-sitter-cpp.git
git clone https://github.com/tree-sitter/tree-sitter-java.git
git clone https://github.com/tree-sitter/tree-sitter-python.git
cd ..

# Evosuite
cd codegen_sources/test_generation/
wget https://github.com/EvoSuite/evosuite/releases/download/v1.1.0/evosuite-1.1.0.jar
cd ../..

# Apex (needs Nvidia GPU)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

# JavaFX SDK 11
# For other platforms visit https://gluonhq.com/products/javafx/ and select 'Include older versions' to download SDK
wget https://download2.gluonhq.com/openjfx/11/openjfx-11_linux-x64_bin-sdk.zip
unzip openjfx-11_linux-x64_bin-sdk.zip

# TransCoder and TransCoder-ST models
mkdir models
cd models
wget https://dl.fbaipublicfiles.com/transcoder/pre_trained_models/TransCoder_model_1.pth
wget https://dl.fbaipublicfiles.com/transcoder/pre_trained_models/TransCoder_model_2.pth
wget https://dl.fbaipublicfiles.com/transcoder/pre_trained_models/online_st_models/Online_ST_CPP_Java.pth
wget https://dl.fbaipublicfiles.com/transcoder/pre_trained_models/online_st_models/Online_ST_CPP_Python.pth
wget https://dl.fbaipublicfiles.com/transcoder/pre_trained_models/online_st_models/Online_ST_Java_CPP.pth
wget https://dl.fbaipublicfiles.com/transcoder/pre_trained_models/online_st_models/Online_ST_Java_Python.pth
wget https://dl.fbaipublicfiles.com/transcoder/pre_trained_models/online_st_models/Online_ST_Python_CPP.pth
wget https://dl.fbaipublicfiles.com/transcoder/pre_trained_models/online_st_models/Online_ST_Python_Java.pth
cd ..

# Dataset
cd data
wget https://dl.fbaipublicfiles.com/transcoder/test_set/transcoder_test_set.zip
unzip transcoder_test_set.zip
cd ..