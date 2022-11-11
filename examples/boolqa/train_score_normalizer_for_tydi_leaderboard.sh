# Train the score normalizer using the dev set split in two. 
# This uses the tydi google preprocessor to generate the appropriate format for the TyDi leaderboard submission.

output_dir="<Location to store output>"
model_dir='<local model or model on HF hub'
data_dir='<location of TyDi dev split in two files'

echo "========"
echo "STEP 1: run mrc and qtc on split 0"
echo "========"
python primeqa/mrc/run_mrc.py --model_name_or_path $model_dir \
       --eval_file ${data_dir}/tydiqa-v1.0-dev-00.jsonl.gz \
       --output_dir ${output_dir}/mrc-dev00 --fp16 \
       --do_eval \
       --per_device_eval_batch_size 128 \
       --overwrite_output_dir \
       --overwrite_cache \
       --preprocessor primeqa.mrc.processors.preprocessors.tydiqa_google.TyDiQAGooglePreprocessor \
       --postprocessor primeqa.boolqa.processors.postprocessors.extractive.ExtractivePipelinePostProcessor \
       --preprocessing_num_workers 10

python primeqa/text_classification/run_nway_classifier.py \
       --model_name_or_path "<QTC model>" \
       --example_id_key example_id --sentence1_key question \
       --label_list "boolean", "other" \
       --output_label_prefix question_type \
       --use_auth_token true \
       --test_file ${output_dir}/mrc-dev00/eval_predictions.json \
       --output_dir ${output_dir}/mrc-dev00/qtc/ \
       --do_predict \
       --per_device_eval_batch_size 128 \
       --overwrite_output_dir \
       --overwrite_cache \
       --do_mrc_pipeline \

echo "========"
echo "STEP 2: train score normalizer with split 0"
echo "========"
python primeqa/boolqa/run_score_normalizer.py \
        --do_train \
        --train_file ${output_dir}/mrc-dev00/qtc/predictions.json \
        --gold_file ${data_dir}/tydiqa-v1.0-dev-00.jsonl.gz \
        --output_dir ${output_dir} \
        --google_format

echo "========"
echo "STEP 3: run boolean mrc on split 1"
echo "========"
python primeqa/mrc/run_mrc.py --model_name_or_path $model_dir \
       --eval_file ${data_dir}/tydiqa-v1.0-dev-01.jsonl.gz \
       --output_dir ${output_dir}/mrc-dev01 --fp16 \
       --do_eval \
       --per_device_eval_batch_size 128 \
       --do_boolean --boolean_config  primeqa/boolqa/tydi_boolqa_config.json \
       --overwrite_output_dir \
       --overwrite_cache \
       --preprocessor primeqa.mrc.processors.preprocessors.tydiqa_google.TyDiQAGooglePreprocessor \
       --preprocessing_num_workers 10