## Description

Instructions for running question answering when the answer is known to be a list. The first part processes the NQ dataset to get the list subset and convert into the TyDi Google format. A short answer model fine-tuned on list answers is ideal to achieve good performance. 

## Get the NQ train and dev list subsets from the full NQ dataset and convert to TyDi Google format (TyDiQAGooglePreprocessor):
```

TRAIN_INPUT_FILE = /location/of/NQ/dev/set/file(s) use * to read multiple files
TRAIN_OUTPUT_FILE = /location/to/save/output.jsonl

OneQA/examples/listqa/list_nq2tydi.py
        --input_file $TRAIN_INPUT_FILE
        --output_file $TRAIN_OUTPUT_FILE

EVAL_INPUT_FILE = /location/of/NQ/train/set/file(s) use * to read multiple files
EVAL_OUTPUT_FILE = /location/to/save/output.jsonl

OneQA/examples/listqa/list_nq2tydi.py
        --input_file $INPUT_FILE
        --output_file $OUTPUT_FILE
```

## Run Command to Train and Evaluate on List Data:
For listqa the answer length must be longer (1000) and there are less annotations so the non-null threshold must be 1.


```
MODEL_DIR = xlm-roberta or roberta-large or fine-tuned QA model
TRAIN_OUTPUT_FILE and EVAL_OUTPUT_FILE are OUTPUT_FILE(s) from previous step
OUTPUT_DIR = /location/to/save/output

OneQA/examples/mrc/run_mrc.py
       --model_name_or_path ${MODEL_DIR} \
       --train_file ${TRAIN_OUTPUT_FILE} \
       --eval_file  ${EVAL_OUTPUT_FILE}\
       --do_train \
       --do_eval \
       --output_dir ${OUTPUT_DIR}/output \
       --max_seq_length 512 \
       --doc_stride 256 \
       --gradient_accumulation_steps 8 \
       --learning_rate 5e-05 \
       --max_answer_length 1000 \
       --per_device_eval_batch_size 64 \
       --per_device_train_batch_size 16 \
       --num_train_epochs 1 \
       --preprocessor oneqa.mrc.processors.preprocessors.tydiqa_google.TyDiQAGooglePreprocessor \
       --passage_non_null_threshold 1 \
       --minimal_non_null_threshold 1 \

```
