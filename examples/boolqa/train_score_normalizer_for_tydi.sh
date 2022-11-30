# Assumes the output directory exists
output_dir="score_normalizer_train_output/"

# This script reads the tydi data from HF and splits the train file:
# 10% of tydi train becomes train_dev, output at ${output_dir}/tydi_train_dev.json
# 90% of tydi train becomes train_train, output at ${output_dir}/tydi_train_train.json
python examples/boolqa/score_normalizer_data_for_tydi.py --output_dir $output_dir

# Run mrc: train on ${output_dir}/tydi_train_train.json
#          decode on ${output_dir}/tydi_train_dev.json
# This produces a file used to create the score normalizer features
# ${output_dir}/mrc/evc/eval_predictions.json
# Note that features from the QTC and EVC are already included in the output file
# --do_boolean --boolean_config  primeqa/boolqa/tydi_boolqa_config.json 
python primeqa/mrc/run_mrc.py --model_name_or_path xlm-roberta-large \
       --train_file ${output_dir}/tydi_train_train.json \
       --eval_file ${output_dir}/tydi_train_dev.json \
       --output_dir ${output_dir}/mrc --fp16 \
       --do_train --do_eval \
       --per_device_train_batch_size 32 --gradient_accumulation_steps 2 \
       --per_device_eval_batch_size 128 \
       --warmup_ratio 0.1 --weight_decay 0.1 --save_steps 50000 \
       --learning_rate 4e-5 \
       --num_train_epochs 1 \
       --do_boolean --boolean_config  primeqa/boolqa/tydi_boolqa_config.json \
       --overwrite_output_dir \
       --overwrite_cache

# Training the score normalizer
# --train_file: the MRC output file for the 10% of tydi train
# --gold file: the 10% of the tydi train in the original format
# the score normalizer is saved at ${output_dir}/score_normalizer_svm.pickle
# you can update the boolq config file with the new model
python primeqa/boolqa/run_score_normalizer.py \
        --do_train \
        --train_file ${output_dir}/mrc/evc/eval_predictions.json \
        --gold_file ${output_dir}/tydi_train_dev.json \
        --output_dir ${output_dir}            