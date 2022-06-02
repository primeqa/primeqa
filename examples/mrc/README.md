# Machine Reading Comprehension (MRC)

The primary script is [run_mrc.py](./run_mrc.py).  This runs a transformer-based MRC pipeline.
Before continuing below make sure you have OneQA [installed](../../README.md#Installation).

## Supported Datasets
Currently supported datasets include:
- TyDiQA
- SQUAD

## Example Usage
An example usage for train + eval command on the TyDiQA dataset (default) is:
```shell
python examples/mrc/run_mrc.py --model_name_or_path xlm-roberta-large \
       --output_dir ${OUTPUT_DIR} --fp16 --learning_rate 4e-5 \
       --do_train --do_eval --per_device_train_batch_size 16 \
       --per_device_eval_batch_size 128 --gradient_accumulation_steps 4 \
       --warmup_ratio 0.1 --weight_decay 0.1 --save_steps 50000 \
       --overwrite_output_dir --num_train_epochs 1 --evaluation_strategy no
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

For eval-only with support for boolean questions (for [details](../boolqa/README.md)):
```shell
python examples/mrc/run_mrc.py --model_name_or_path ${BOOLEAN_MODEL_NAME} \
       --output_dir ${OUTPUT_DIR} --fp16 --do_eval \
       --do_boolean \
       --per_device_eval_batch_size 128 --overwrite_output_dir
```



For eval with confidence calibration, add the following additional command line arguments:
```shell
      --output_dropout_rate 0.25 \
       --decoding_times_with_dropout 5 \
       --confidence_model_path ${CONFIDENCE_MODEL_PATH} \
       --task_heads oneqa.mrc.models.heads.extractive.EXTRACTIVE_WITH_CONFIDENCE_HEAD
```

For the SQUAD dataset use the folowing additional command line arguments for train + eval :
```shell
       --dataset_name squad \
       --dataset_config_name plain_text \
       --preprocessor oneqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
       --postprocessor oneqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
       --eval_metrics squad 
```

This will detect a GPU if present as well as multiple CPU cores for accelerating preprocessing.
Some hyperparameters (e.g. fp16, batch size, gradient accumulation steps) may need to be changed
depending on your hardware configuration.

The dataset name and config are currently omitted as only TyDi is supported at the moment.

### Task Arguments

Some task arguments take references which allow for dynamic imports of existing or
user-defined functionality.  For example, to select the `ExtractivePostProcessor` use
`--postprocessor oneqa.mrc.processors.postprocessors.extractive.ExtractivePostProcessor`.
Alternatively, a new postprocessor could be written and selected with 
`--postprocessor qualified.path.to.new.postprocessor.NewPostProcessor`.

For example, if one was implementing a new model which made predictions by means other than
an extractive head then a `NewPostProcessor` which derived predictions from the model
outputs would be needed.

Similarly, when adding support for a new dataset (with a new schema) a new preprocessor would be needed.
This would be selected by specifying `--preprocessor qualified.path.to.new.postprocessor.NewPreProcessor`
for the `NewPreProcessor` corresponding to this dataset and schema.
