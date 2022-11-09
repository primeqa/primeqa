input_released_data=${1:-data/released_data/test.json}
processed_input_path=${2:-data/processed_data/test_processed.json}
rr_config_path=${3:-data/row_ret_config/predict_test_no_group_ranked_passage_large.json}
rr_pred_path=${4:-data/predictions/row_retriever/on_test_BLarge_no_group_ranked_passage.json}
answer_extractor_input_file_path=${5:-data/processed_data/input_for_answer_extractor.json}
answer_extractor_model_path=${6:-/dccstor/cssblr/vishwajeet/git/tabtextqa/rc_bert_large_pl_409/2021_09_04_06_47_55/checkpoint-epoch2/}
answer_extractor_output_dir=${7:-data/predictions/answer_extractor/}
final_prediction_file_path=${8:-data/predictions/answer_extractor/test_pred.json}

#python passage_filtering_based_on_sentence_transformers.py $input_released_data $processed_input_path

# python row_retriever_MITQA.py \
# --config $rr_config_path \
# --test

python process_row_retriever_output.py \
 --released_data_path $input_released_data \
 --row_ret_pred_path $rr_pred_path \
 --processed_output_file_path $answer_extractor_input_file_path

python3 answer_extractor.py --model_type bert \
                              --model_name_or_path $answer_extractor_model_path \
                              --do_stage3 --do_lower_case \
                              --predict_file $answer_extractor_input_file_path \
                              --per_gpu_train_batch_size 8 \
                              --max_seq_length 512 \
                              --doc_stride 128 \
                              --threads 8 \
                              --output_dir $answer_extractor_output_dir \
                              --pred_ans_file $final_prediction_file_path \
                              --num_train_epochs 5 \


python re-rank_answer_extractor_output.py \
 data/predictions/answer_extractor/test_pred.json data/predictions/answer_extractor/test_pred_re-ranked.json

python utils/convert_to_challenge_format.py \
  data/predictions/answer_extractor/test_pred_re-ranked.json data/predictions/answer_extractor/test_answers.json

