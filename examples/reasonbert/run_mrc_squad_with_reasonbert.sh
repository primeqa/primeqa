MODEL=osunlp/ReasonBERT-RoBERTa-base  # or roberta-base for baseline
MRQA_SUBSET=SQuAD  # one of SQuAD NaturalQuestionsShort NewsQA TriviaQA-web SearchQA
OUTPUT_DIR=output_${MRQA_SUBSET}
NUM_EPOCHS=10
LR=5e-5
MSL=512    # 384 for SQuAD
STRIDE=384 # 128 for SQuAD
MSL=384
STRIDE=128
NUM_TRAIN_SAMPLES=128
SEED=234

python primeqa/mrc/run_mrc.py \
    --model_name_or_path ${MODEL} \
    --dataset_name mrqa --max_seq_length ${MSL} --doc_stride ${STRIDE} \
    --dataset_config_name plain_text \
    --dataset_filter_column_values ${MRQA_SUBSET} \
    --dataset_filter_column_name subset \
    --max_train_samples ${NUM_TRAIN_SAMPLES} --seed ${SEED} \
    --learning_rate ${LR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --preprocessor primeqa.mrc.processors.preprocessors.mrqa.MRQAPreprocessor \
    --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
    --negative_sampling_prob_when_has_answer 1.0 \
    --negative_sampling_prob_when_no_answer 1.0 \
    --eval_metrics SQUAD \
    --output_dir ${OUTPUT_DIR} \
    --do_train --do_eval \
    --fp16 \
    --per_device_train_batch_size 20 --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 --warmup_ratio 0.1 --weight_decay 0.01 \
    --save_steps 50000  \
    --evaluation_strategy no --single_context_multiple_passages