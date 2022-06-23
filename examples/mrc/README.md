# Machine Reading Comprehension (MRC)

The primary script is [run_mrc.py](./run_mrc.py).  This runs a transformer-based MRC pipeline.
Before continuing below make sure you have PrimeQA [installed](../../README.md#Installation).

## Supported Datasets
Currently supported datasets include:
- TyDiQA
- SQuAD 1.1
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
       --overwrite_output_dir --num_train_epochs 1 --evaluation_strategy no
```
This will detect a GPU if present as well as multiple CPU cores for accelerating preprocessing.
Some hyperparameters (e.g. fp16, batch size, gradient accumulation steps) may need to be changed
depending on your hardware configuration.

This yields the following results \TODO:
```add our results here in the HF format as shown in https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering
```

For just training:
```shell
python examples/mrc/run_mrc.py --model_name_or_path xlm-roberta-large \
       --output_dir ${TRAINING_OUTPUT_DIR} --fp16 --learning_rate 4e-5 \
       --do_train --per_device_train_batch_size 16 --gradient_accumulation_steps 4 \
       --warmup_ratio 0.1 --weight_decay 0.1 --save_steps 50000 \
       --overwrite_output_dir --num_train_epochs 1 --evaluation_strategy no
```

For just eval:
```shell
python examples/mrc/run_mrc.py --model_name_or_path ${TRAINING_OUTPUT_DIR} \
       --output_dir ${OUTPUT_DIR} --fp16 --do_eval \
       --per_device_eval_batch_size 128 --overwrite_output_dir
```
- if you want to do [calbrate the confidence](https://arxiv.org/abs/2101.07942) estimate of your fine-tuned model use the following:

For eval with confidence calibration, add the following additional command line arguments:
```shell
      --output_dropout_rate 0.25 \
       --decoding_times_with_dropout 5 \
       --confidence_model_path ${CONFIDENCE_MODEL_PATH} \
       --task_heads primeqa.mrc.models.heads.extractive.EXTRACTIVE_WITH_CONFIDENCE_HEAD
```
This yields the following results \TODO:
```add our results here in the HF format as shown in https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering
```

 - Dataset: [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)

For the SQUAD 1.1 dataset use the folowing additional command line arguments for train + eval :
```shell
       --dataset_name squad \
       --dataset_config_name plain_text \
       --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
       --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
       --eval_metrics squad 
```
This yields the following results \TODO:
```add our results here in the HF format as shown in https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering
```

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
This yields the following results \TODO:
```add our results here in the HF format as shown in https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering
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
