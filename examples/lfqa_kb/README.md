# Long-form Question Answering

The code is implemented with Huggingface 4.17. The pipeline is modified from [this Transformers example](https://github.com/huggingface/transformers/blob/v4.17.0/examples/pytorch/question-answering/run_seq2seq_qa.py)

## Requirements
- primeqa
- datasets

## Training
You can train the model with the following scripts.

### To Train BART
You can run the script with:
```sh
output_dir=<your_output_dir>

jbsub -q x86_24h -cores 1+1 -mem 32g -require 'v100' -proj 'eli5' -name 'train_large_beam' -o ${output_dir}/train.log \
python /dccstor/myu/OneQA/examples/lfqa_kb/run_lfqa.py \
  --model_name_or_path facebook/bart-large \
  --train_file <your_data_dir>/train-kilt-dpr.json \
  --validation_file <your_data_dir>/eli5-dev-kilt-dpr.json \
  --question_column input \
  --answer_column output \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 64 \
  --generation_max_length 256 \
  --max_answer_length 256 \
  --fp16 \
  --predict_with_generate \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --load_best_model_at_end \
  --metric_for_best_model rougeL \
  --num_beams 4 \
```

### To Train DPR+BART
```sh
output_dir=<your_output_dir>

jbsub -q x86_24h -cores 1+1 -mem 40g -require 'v100' -proj 'eli5' -name 'train_dprbart' -o ${output_dir}/train.log \
python /dccstor/myu/OneQA/examples/lfqa_kb/run_lfqa.py \
  --model_name_or_path facebook/bart-large \
  --train_file <your_data_dir>/eli5-train-kilt-dpr.json \
  --validation_file <your_data_dir>/eli5-dev-kilt-dpr.json \
  --question_column input \
  --answer_column output \
  --context_column passages \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 512 \
  --generation_max_length 256 \
  --max_answer_length 256 \
  --fp16 \
  --predict_with_generate \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --metric_for_best_model rougeL \
  --num_beams 4 \
  --n_context 3 \
```

### To Train FiD
```sh
output_dir=<your_output_dir>

jbsub -q x86_24h -cores 1+1 -mem 40g -require 'v100' -proj 'eli5' -name 'train_fid' -o ${output_dir}/train.log \
python /dccstor/myu/OneQA/examples/lfqa_kb/run_fid.py \
  --model_name_or_path facebook/bart-large \
  --train_file <your_data_dir>/eli5-train-kilt-dpr.json \
  --validation_file <your_data_dir>/eli5-dev-kilt-dpr.json \
  --question_column input \
  --answer_column output \
  --context_column passages \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-5 \
  --num_train_epochs 4 \
  --max_seq_length 256 \
  --max_answer_length 256 \
  --generation_max_length 256 \
  --fp16 \
  --predict_with_generate \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --metric_for_best_model rougeL \
  --save_total_limit 2 \
  --n_context 3 \
  --num_beams 4 \
```

# Evaluation
You can also do evaluation from a trained checkpoint.

For example:
```sh
model_dir=<yout_ckpt_dir>/checkpoint-32904

python /dccstor/myu/OneQA/examples/lfqa_kb/run_fid.py \
  --model_name_or_path ${model_dir} \
  --train_file <your_data_dir>/eli5-train-kilt-dpr.json \
  --validation_file <your_data_dir>/eli5-dev-kilt-dpr.json \
  --question_column input \
  --answer_column output \
  --context_column passages \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --max_seq_length 256 \
  --max_answer_length 256 \
  --generation_max_length 256 \
  --fp16 \
  --predict_with_generate \
  --output_dir ${model_dir}/predictions \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --metric_for_best_model rougeL \
  --save_total_limit 2 \
  --n_context 3 \
  --num_beams 4 \
```