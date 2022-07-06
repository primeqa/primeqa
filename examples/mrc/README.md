# Machine Reading Comprehension (MRC)

The primary script is [run_mrc.py](./run_mrc.py).  This runs a transformer-based MRC pipeline.
Before continuing below make sure you have PrimeQA [installed](../../README.md#Installation).

## Supported Datasets
Currently supported datasets include:
- TyDiQA
- SQuAD 1.1
- XQuAD
- MLQA

## Example Usage

 - Dataset: [TyDiQA](https://ai.google.com/research/tydiqa)

An example usage for train + eval command on the TyDiQA dataset (default) is:
```shell
python examples/mrc/run_mrc.py --model_name_or_path xlm-roberta-large \
       --output_dir ${OUTPUT_DIR} --fp16 --learning_rate 4e-5 \
       --do_train --do_eval --per_device_train_batch_size 16 \
       --per_device_eval_batch_size 128 --gradient_accumulation_steps 4 \
       --warmup_ratio 0.1 --weight_decay 0.1 --save_steps 50000 \
       --overwrite_output_dir --num_train_epochs 1 
       --evaluation_strategy no --overwrite_cache
```
This will detect a GPU if present as well as multiple CPU cores for accelerating preprocessing.
Some hyperparameters (e.g. fp16, batch size, gradient accumulation steps) may need to be changed
depending on your hardware configuration.

The trained model is available [here](https://huggingface.co/ibm/tydiqa-primary-task-xlm-roberta-large).

This yields the following results:
```
***** eval metrics *****
epoch = 1.0
eval_avg_minimal_f1 = 0.6745
eval_avg_minimal_precision = 0.7331
eval_avg_minimal_recall = 0.628
eval_avg_passage_f1 = 0.7215
eval_avg_passage_precision = 0.7403
eval_avg_passage_recall = 0.7061
eval_samples = 18670
```

For just training:
```shell
python examples/mrc/run_mrc.py --model_name_or_path xlm-roberta-large \
       --output_dir ${TRAINING_OUTPUT_DIR} --fp16 --learning_rate 4e-5 \
       --do_train --per_device_train_batch_size 16 --gradient_accumulation_steps 4 \
       --warmup_ratio 0.1 --weight_decay 0.1 --save_steps 50000 \
       --overwrite_output_dir --num_train_epochs 1 
       --evaluation_strategy no --overwrite_cache
```

For just eval:
```shell
python examples/mrc/run_mrc.py --model_name_or_path ${TRAINING_OUTPUT_DIR} \
       --output_dir ${OUTPUT_DIR} --fp16 --do_eval \
       --per_device_eval_batch_size 128 --overwrite_output_dir --overwrite_cache
```

- if you want to do [confidence calibration](https://arxiv.org/abs/2101.07942) estimate of your fine-tuned model use the following:


For eval with confidence calibration, add the following additional command line arguments:
```shell
       --output_dropout_rate 0.25 \
       --decoding_times_with_dropout 5 \
       --confidence_model_path ${CONFIDENCE_MODEL_PATH} \
       --task_heads primeqa.mrc.models.heads.extractive.EXTRACTIVE_WITH_CONFIDENCE_HEAD
```
 - Dataset: [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)

For the SQUAD 1.1 dataset use the folowing additional command line arguments for train + eval :
```shell
       --dataset_name squad \
       --dataset_config_name plain_text \
       --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
       --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
       --eval_metrics SQUAD 
```
This yields the following results:
```
***** eval metrics ***** 
eval_exact_match = 88.7133
eval_f1          = 94.3525
```
 - Dataset: [XQuAD](https://arxiv.org/pdf/1910.11856v3.pdf)


For the XQuAD dataset run the evaluation script after the model has been trained on SQuAD 1.1. 
The dataset configurations for all languages are supported.
For the XQuAD in ZH use the following command line arguments for eval:
```shell
       --dataset_name xquad \
       --dataset_config_name xquad.zh \
       --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
       --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
       --eval_metrics SQUAD 
```
This yields the following results:

|  | en   | es   |  de  |  el |  ru  |  tr | ar  | vi  | th | zh | hi |
|--| ---- | -----|------|-----|------|-----|-----|-----|----|----|----|
|F1| 87.5 | 82.1 | 80.7 |81.5 | 80.0 | 75.0| 75.1| 80.0|75.3|70.3|77.2|
|EM| 76.7 | 63.4 | 65.4 |64.2 | 63.6 | 59.3| 59.1| 61.3|65.5|62.2|61.8|

 - Dataset: [MLQA](https://github.com/facebookresearch/MLQA)

For the MLQA dataset run the evaluation script after the model has been trained on SQuAD 1.1. 
The dataset configurations for all language combinations are supported.
For the MLQA configuration with context language EN and question language DE use the following command line arguments for eval:
```shell
       --dataset_name mlqa \
       --dataset_config_name mlqa.en.de \
       --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
       --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
       --eval_metrics MLQA 
```
This yields the following results:

|  | en   | es   |  de  |  ar |  hi  |  vi | zh  |
|--| ---- | -----|------|-----|------|-----|-----|
|F1| 84.8 | 75.9 | 68.8 |67.7 | 72.1 | 71.8| 69.8|
|EM| 72.9 | 57.2 | 52.7 |46.6 | 55.6 | 52.1| 50.0|

 -  PrimeQA also supports special Features for MRC systems as follows:

 -  Answering [Boolean Questions](https://arxiv.org/abs/1905.10044) for TyDI (currently in an inference-only setup). Please read the [details](../boolqa/README.md)):
```shell
python examples/mrc/run_mrc.py --model_name_or_path PrimeQA/tydiqa-primary-task-xlm-roberta-large \
       --output_dir ${OUTPUT_DIR} --fp16 --overwrite_cache \
       --per_device_eval_batch_size 128 --overwrite_output_dir \
       --do_boolean --boolean_config  examples/boolqa/tydi_boolqa_config.json
```
The corresponding model files are available as part of these: [Question classifier](https://huggingface.co/PrimeQA/tydiqa-boolean-question-classifier), [Answer classifier](https://huggingface.co/PrimeQA/tydiqa-boolean-answer-classifier), [MRC system](https://huggingface.co/PrimeQA/tydiqa-primary-task-xlm-roberta-large). This setup is based on the top submission to the minimal answer leaderboard (hidden blind test) for TyDI (as of 7/2/2022).

This yields the following results:
```
***** eval metrics *****
epoch = 1.0
eval_avg_minimal_f1 = 0.7151
eval_avg_minimal_precision = 0.7229
eval_avg_minimal_recall = 0.7097
eval_avg_passage_f1 = 0.7447
eval_avg_passage_precision = 0.7496
eval_avg_passage_recall = 0.7433
eval_samples = 18670
```

 - PrimeQA also supports answering questions to which answers are collective e.g. lists.

For Training/Evaluating questions with lists as answers it is important to include the following argument parameters and values. The answer length must be longer and there are less annotations so the non-null threshold must be 1 (There are no null answers). See `examples/listqa/README.md` for more information and a use case using NQ list data:
```
       --max_seq_length 512 \
       --learning_rate 5e-05 \
       --max_answer_length 1000 \
       --passage_non_null_threshold 1 \
       --minimal_non_null_threshold 1 \
```
This yields the following results on English only using the TyDi evaluation script with two training strategies:
```
xlm-roberta-large -> NQ Lists: Minimal F1 = 46.95
xlm-roberta-large -> PrimeQA/tydiqa-primary-task-xlm-roberta-large -> NQ Lists: Minimal F1 = 57.44
```


### Task Arguments

Some task arguments take references which allow for dynamic imports of existing or
user-defined functionality.  For example, to select the `ExtractivePostProcessor` use
`--postprocessor primeqa.mrc.processors.postprocessors.extractive.ExtractivePostProcessor`.
Alternatively, a new postprocessor could be written and selected with 
`--postprocessor qualified.path.to.new.postprocessor.NewPostProcessor`.

For example, if one was implementing a new model which made predictions by means other than
an extractive head then a `NewPostProcessor` which derived predictions from the model
outputs would be needed.

Similarly, when adding support for a new dataset (with a new schema) a new preprocessor would be needed.
This would be selected by specifying `--preprocessor qualified.path.to.new.postprocessor.NewPreProcessor`
for the `NewPreProcessor` corresponding to this dataset and schema.
