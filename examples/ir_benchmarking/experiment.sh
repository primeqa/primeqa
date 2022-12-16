
export JAVA_HOME=jdk-11.0.1
#export PATH=$JAVA_HOME/bin:$PATH
export PATH="/usr/local/cuda-12.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH"
#export CUDA_VISIBLE_DEVICES="0"
export CUDA_HOME=/usr/local/cuda-12.0

python index_and_evaluate_msmarco_eacl.py

#python run_ir.py \
#    --engine_type DPR \
#    --do_search \
#    --queries xorqa_dev_gmt.tsv \
#    --model_name_or_path XORTyDi_En_DPR_model/qry_encoder/ \
#    --qry_tokenizer_path facebook/dpr-question_encoder-multiset-base \
#    --bsize 32 \
#    --index_location DPR/XORTyDi_En_DPR_index/ \
#    --output_dir ./ \
#    --top_k 10 \
#    --n_gpu 1
