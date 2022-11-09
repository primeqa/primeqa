train_path=${1:-processed_train_data_answer_extractor.json}
dev_path=${2:-processed_dev_data_answer_extracton.json}
output_dir=${3:-./answer_extractor_model/}
pred_ans_path=${4:-./answer_extractor_model/pred_dev.json}


python3 answer_extractor.py --model_type bert \
                              --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
                              --do_train --do_stage3 --do_lower_case \
                              --train_file $train_path \
                              --predict_file $dev_path \
                              --per_gpu_train_batch_size 8 \
                              --max_seq_length 512 \
                              --doc_stride 128 \
                              --threads 8 \
                              --output_dir $output_dir \
                              --pred_ans_file $pred_ans_path \
                              --num_train_epochs 5 \
                          