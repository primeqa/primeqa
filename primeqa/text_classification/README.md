# Text Classification

The run_nway_classifier can be used to train classify one or two spans of text with n labels supplied at runtime. This classifier is a modification of [run_glue.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) on HuggingFace. We provide example usage below.

The text that is classified is considered sentence 1 and sentence 2. The keys for this data and the ids in the input file are provided as args `sentence1_key`, `sentence2_key`, and `id_key`. The labels are provided in the `label_list` arg.

## Question type classification

Given a question, predict its type, e.g. whether the question is `boolean` or `other` (e.g. short answer, table answer, list answer). 

```shell
python primeqa/text_classification/run_nway_classifier.py \
    --overwrite_cache \
    --id_key example_id \
    --sentence1_key question \
    --label_list other boolean \
    --output_label_prefix question_type \
    --model_name_or_path PrimeQA/qtc_bert_pretrained_model \
    --test_file ${BASE}/eval_predictions.json \
    --output_dir ${OUTDIR}/qtc \
    --use_auth_token
```

## Answer classification

Given a question and a passage predict whether the span supports
a `yes`, `no`, or `no answer` answer to the question.

```shell
python primeqa/boolqa/run_nway_classifier.py \
    --overwrite_cache  \
    --id_key example_id \
    --sentence1_key question \
    --sentence2_key passage_answer_text \
    --label_list no no_answer yes \
    --output_label_prefix boolean_answer \
    --drop_label no_answer \
    --model_name_or_path PrimeQA/evc_xlm_roberta_large \
    --test_file ${BASE}/qtc/eval_predictions.json \
    --output_dir ${OUTDIR}/evc \
    --use_auth_token
```
