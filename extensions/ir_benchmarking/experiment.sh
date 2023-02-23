
export JAVA_HOME=jdk-11.0.1
#export PATH=$JAVA_HOME/bin:$PATH
export PATH="/usr/local/cuda-12.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES="0"
export CUDA_HOME=/usr/local/cuda-12.0

python plaid_colbertv2_evaluation.py
