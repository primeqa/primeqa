output_dir="/dccstor/mabornea1/primeqa_code/boolean_train/exp_test/"

python primeqa/boolqa/score_normalizer_data_for_tydi.py --output_dir $output_dir

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

python primeqa/boolqa/run_score_normalizer.py \
        --train_file ${output_dir}/mrc/evc/eval_predictions.json \
        --gold_file ${output_dir}/tydi_train_dev.json \
        --do_train \
        --output_dir ${output_dir}            