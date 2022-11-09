model_path=${1:-data/trained_models/answer_extractor/rc_bert_large_logits_top_pl_min/2021_09_01_03_14_50/checkpoint-epoch3/}
dev_path=${2:-data/processed_data/input_for_answer_extraction_test_split.json}
output_dir=${3:-data/predictions/answer_extractor/}
pred_ans_path=${4:-data/predictions/answer_extractor/pred_test.json}


python3 answer_extractor.py --model_type bert \
                              --model_name_or_path $model_path \
                              --do_stage3 --do_lower_case \
                              --predict_file $dev_path \
                              --per_gpu_train_batch_size 8 \
                              --max_seq_length 512 \
                              --doc_stride 128 \
                              --threads 8 \
                              --output_dir $output_dir \
                              --pred_ans_file $pred_ans_path \
                              --num_train_epochs 5 \
                          