<!-- START sphinx doc instructions - DO NOT MODIFY next code, please -->
<details>
<summary>API Reference</summary>    

```{eval-rst}

.. autosummary::
    :toctree: _autosummary
    :template: custom-module-template.rst
    :recursive:
   
    primeqa.mrc

```
</details>          
<br>
<!-- END sphinx doc instructions - DO NOT MODIFY above code, please --> 

# Machine Reading Comprehension (MRC)

Before continuing below make sure you have PrimeQA [installed](https://primeqa.github.io/primeqa/installation.html).

## Inference Example Usage
The following shows how to use the MRC component within PrimeQA to extract an answer given a question and a context:

 - Step 1:  Initialize the reader. You can choose any of the MRC models we currently have [here](https://huggingface.co/PrimeQA).
```python
import json
from primeqa.components.reader.extractive import ExtractiveReader
reader = ExtractiveReader("PrimeQA/nq_tydi_sq1-reader-xlmr_large-20221110")
reader.load()
```
- Step 2: Execute the reader in inference mode:
```python
question = ["Which country is Canberra located in?"]
context = [["""Canberra is the capital city of Australia. 
Founded following the federation of the colonies of Australia 
as the seat of government for the new nation, it is Australia's 
largest inland city"""]]
answers = reader.predict(question,context)  
print(json.dumps(answers, indent=4))  
```
The above statements will generate an output in the form of a dictionary:
```shell
"0": [
        {
            "example_id": "0",
            "passage_index": 0,
            "span_answer_text": "Australia",
            "span_answer": {
                "start_position": 32,
                "end_position": 41
            },
            "span_answer_score": 14.109326839447021,
            "confidence_score": 0.6732346778531001
        },
        {
            "example_id": "0",
            "passage_index": 0,
            "span_answer_text": "Australia. \nFounded following the federation of the colonies of Australia",
            "span_answer": {
                "start_position": 32,
                "end_position": 105
            },
            "span_answer_score": 12.882871329784393,
            "confidence_score": 0.1974802270822016
        },
        {
            "example_id": "0",
            "passage_index": 0,
            "span_answer_text": "Australia. \nFounded following the federation of the colonies of Australia \nas the seat of government for the new nation, it is Australia",
            "span_answer": {
                "start_position": 32,
                "end_position": 168
            },
            "span_answer_score": 12.459252871572971,
            "confidence_score": 0.12928509506469837
        }
    ]
}
```

Additional inference examples can be found in the python [notebook](../../notebooks/mrc/mrc_usage_predict_mode.ipynb).

## Train and Evaluate
If you want to perform a fully functional train and inference procedure for the MRC components, then the primary script to use is [run_mrc.py](https://github.com/primeqa/primeqa/blob/main/primeqa/mrc/run_mrc.py).  This runs a transformer-based MRC pipeline.

### Supported Datasets
Currently supported MRC datasets include:
- TyDiQA
- SQuAD 1.1
- XQuAD
- MLQA
- Natural Questions(NQ)
- Custom Data
- MRQA

Currently supported TableQA datasets :
- WikiSQL
- SQA

User's can also provide their own [custom data](#custom-data) 

### Example Usage

 #### [TyDiQA](https://ai.google.com/research/tydiqa)

An example usage for train + eval command on the TyDiQA dataset (default) is:
```shell
python primeqa/mrc/run_mrc.py --model_name_or_path xlm-roberta-large \
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
python primeqa/mrc/run_mrc.py --model_name_or_path xlm-roberta-large \
       --output_dir ${TRAINING_OUTPUT_DIR} --fp16 --learning_rate 4e-5 \
       --do_train --per_device_train_batch_size 16 --gradient_accumulation_steps 4 \
       --warmup_ratio 0.1 --weight_decay 0.1 --save_steps 50000 \
       --overwrite_output_dir --num_train_epochs 1 
       --evaluation_strategy no --overwrite_cache
```

For just eval:
```shell
python primeqa/mrc/run_mrc.py --model_name_or_path ${TRAINING_OUTPUT_DIR} \
       --output_dir ${OUTPUT_DIR} --fp16 --do_eval \
       --per_device_eval_batch_size 128 --overwrite_output_dir --overwrite_cache
```

 #### [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)

For the SQUAD 1.1 dataset from Huggingface use the folowing additional command line arguments for train + eval :
```shell
       --dataset_name squad \
       --dataset_config_name plain_text \
       --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
       --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
       --eval_metrics SQUAD 
```
For the SQUAD 1.1 dataset in the original Stanford format, first download the train and dev files from here:

- https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
- https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

Use the following additional command line arguments for train and eval:

```
       --train_file <path-to-train-v1.1.json> \
       --eval_file <path-to-dev-v1.1.json> \
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
 #### [XQuAD](https://arxiv.org/pdf/1910.11856v3.pdf)


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

 #### [MLQA](https://github.com/facebookresearch/MLQA)

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

 #### [Natural Questions](https://ai.google.com/research/NaturalQuestions)

For the NQ dataset use the following additional command line arguments for train + eval :
```shell
       --dataset_name natural_questions \
       --dataset_config_name default \
       --postprocessor primeqa.mrc.processors.postprocessors.natural_questions.NaturalQuestionsPostProcessor \
       --preprocessor primeqa.mrc.processors.preprocessors.natural_questions.NaturalQuestionsPreProcessor \
       --beam_runner DirectRunner \
       --num_train_epochs 1 \
       --learning_rate 3e-5 \
       --eval_metrics NQF1
```
This yields the following results:
```
LONG ANSWER R@P TABLE:
Optimal threshold: 4.0808
F1 / P / R
65.09% / 64.54% / 65.65%
R@P=0.5: 77.76% (actual p=50.12%, score threshold=1.979)
R@P=0.75: 41.85% (actual p=75.01%, score threshold=5.523)
R@P=0.9: 4.20% (actual p=90.32%, score threshold=8.189)

SHORT ANSWER R@P TABLE:
Optimal threshold: 4.0822
F1 / P / R
56.76% / 57.24% / 56.28%
R@P=0.5: 61.23% (actual p=50.01%, score threshold=3.235)
R@P=0.75: 29.25% (actual p=75.11%, score threshold=6.031)
R@P=0.9: 10.16% (actual p=90.00%, score threshold=7.425)
```
 #### [MRQA](https://huggingface.co/datasets/mrqa)

 The dataset is a collection of 18 existing QA dataset (carefully selected subset of them) and converted to the same format (SQuAD like format)

 For the MRQA dataset use the following additional command line arguments:
 ```shell
       --dataset_name mrqa \
       --dataset_config_name plain_text \
       --preprocessor primeqa.mrc.processors.preprocessors.mrqa.MRQAPreprocessor \
       --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
       --eval_metrics SQUAD 
```
Additionally, to specify a MRQA subset e.g. `SQuAD`, `NaturalQuestionsShort`, `TriviaQA-web`, use the command line argments  `--dataset_filter_column_name` to specify a column name and `--dataset_filter_column_values` to specify a list of column values.  The example below selects `SQuAD` and `HotpotQA` examples using the column `subset` in the MRQA dataset.  The script `run_mrc.py` shuffles the train examples, eval examples are kept in the same order as read in from the source.
 ```shell
       --dataset_filter_column_values SQuAD HotpotQA
       --dataset_filter_column_name subset
```

Cross domain experiments can be run by running train and eval as separate processes.

 
### Custom Data

Users can also train (fine-tune) and evaluate the MRC model on custom data by providing their own train_file and eval_file. Instructions for getting started are available [here](../../extensions/custom_mrc/README.md).


### Training with Multiple Datasets

PrimeQA supports the training of MRC model with combination of multiple datasets, which are specified in "--train_fof" argument. This argument points to a file of training files (fof). This fof can be in any of three supported formats: csv, jsonl, and json.

In the csv format, each line of train_fof consists of four columns, separated by spaces:
 - HuggingFace dataset name, or path of local data file, or path of dataset processing script (in python);
 - Dataset config name or data file format;
 - Sampling rate within range 0.0 to 1.0, e.g. 0.5 means 50% of the examples are randomly selected and used in MRC training;
 - Preprocessor name.

If column 2 to 4 are not given, default values from input arguments will be used, i.e.:
 - Value of "--dataset_config_name" for HuggingFace dataset and processing script, or "--data_file_format" for local data file;
 - 1.0 for sampling rate;
 - Value of "--preprocessor" for preprocessor name.

If in jsonl format, each line is a dictionary consisting of
```
{'dataset': dataset_name_or_path_of_data_file_or_processing_script,
 'config': dataset_config_or_data_file_format,
 'sampling_rate': sampling_rate,
 'preprocessor': preprocessor_name}
```

Fields 'config', 'sampling_rate', and 'preprocessor' are optional.

If in json format, a list of dictionaries same to that in jsonl is expected.

The following is an example of "--train_fof" in csv format which includes two HuggingFace datatsets: TyDiQA and SQuAD.
```
tydiqa primary_task 0.1 primeqa.mrc.processors.preprocessors.tydiqa.TyDiQAPreprocessor
squad  plain_text   0.1 primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor
```

To evaluate the checkpoint models during training, the validation dataset needs be specified in "--eval_fof" which format is same to "--train_fof". Multiple datatsets can be included into "--eval_fof" if use the trainer 'MSKD_MRCTrainer', otherwise a single validation dataset is expected.

Please note that, if "--train_fof" and "--eval_fof" are given in input arguments, other dataset related parameters, i.e. "--dataset_name", "--train_file", and "--eval_file" are ignored.

The following additional command lines show how to train MRC model with TyDiQA+SQuAD, and evaluate on TyDiQA.
```shell
python primeqa/mrc/run_mrc.py --model_name_or_path xlm-roberta-large \
--train_fof ${train_fof_including_tydi_squad} \
--eval_fof ${eval_fof_including_tydi} \
--postprocessor primeqa.mrc.processors.postprocessors.extractive.ExtractivePostProcessor \
--eval_metrics TyDiF1 \
```


## Special MRC Features:

PrimeQA also supports special features for MRC systems as follows:

### Boolean Questions
Answering [Boolean Questions](https://arxiv.org/abs/1905.10044) for TyDI. Please read the details of [inference](https://primeqa.github.io/primeqa/api/boolqa/index.html) or [training](https://primeqa.github.io/primeqa/extensions/boolqa/index.html):
```shell
python primeqa/mrc/run_mrc.py --model_name_or_path PrimeQA/tydi-reader_bpes-xlmr_large-20221117 \
       --output_dir ${OUTPUT_DIR} --fp16 --overwrite_cache \
       --per_device_eval_batch_size 128 --overwrite_output_dir \
       --do_boolean --boolean_config  extensions/boolqa/tydi_boolqa_config.json
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

### Confidence Calibration

To run [confidence calibration](https://arxiv.org/abs/2101.07942) on your fine-tuned model during inference use the following additional command line arguments:

```shell
       --output_dropout_rate 0.25 \
       --decoding_times_with_dropout 5 \
       --confidence_model_path ${CONFIDENCE_MODEL_PATH} \
       --task_heads primeqa.mrc.models.heads.extractive.EXTRACTIVE_WITH_CONFIDENCE_HEAD
```

### List Answers

PrimeQA also supports answering questions to which answers are collective e.g. lists.

For Training/Evaluating questions with lists as answers it is important to include the following argument parameters and values. The answer length must be longer and there are less annotations so the non-null threshold must be 1 (There are no null answers). See [extensions/listqa/README.md](https://github.com/primeqa/primeqa/blob/main/extensions/listqa/README.md) for more information and a use case using NQ list data:
```
       --max_seq_length 512 \
       --learning_rate 5e-05 \
       --max_answer_length 1000 \
       --passage_non_null_threshold 1 \
       --minimal_non_null_threshold 1 \
```

This yields the following results on English only using the TyDi evaluation script with two training strategies. Please note the ListQA models use the NQ list data by using the long answers offsets as the short answer. Further details can be found in `extensions/listqa/README.md`:

```
xlm-roberta-large -> NQ Lists: Minimal F1 = 47.88
xlm-roberta-large -> PrimeQA/tydiqa-primary-task-xlm-roberta-large -> NQ Lists: Minimal F1 = 58.44 
```

The trained models are available on HuggingFace: [xlm-r->NQ lists](https://huggingface.co/PrimeQA/listqa_nq-task-xlm-roberta-large) and [xlm-r->TyDi->NQ lists](https://huggingface.co/PrimeQA/tydiqa-ft-listqa_nq-task-xlm-roberta-large).

### Table QA
PrimeQA also supports answering questions over tables.

For training and evaluation of a Table Question Answering model on wikisql dataset run the following script:
```shell
       python primeqa/mrc/run_mrc.py --modality "table" \
       --dataset_name "wikisql" \
       --tableqa_config_file "primeqa/tableqa/tableqa_config.json" \
       --output_dir "models/wikisql/" \
       --model_name_or_path "google/tapas-base" \
       --do_train \
       --do_eval
```
This runs a [TAPAS](https://aclanthology.org/2020.acl-main.398.pdf) based tableQA pipeline.

The current performance on wikisql dev set is:
```shell
***** eval metrics *****
Eval denotation accuracy: 86.78%

```
You can also train the tableqa model on your own custom data by proving own train_file and eval_file. Train the TableQA model on custom data using the above script with the following additional parameters:

```shell
       --train_file "<path_to_train.tsv file" \
       --eval_file "<path_to_eval.tsv file" \

```

The format of dataset required for training and evaluation is:

`Question_id\tquestion\ttable_path\tanswer_coordinates\tanswer_text`    

The tables in csv format should be placed under `data_path_root/tables/`. The tables should have first row as column headers.

Our python [notebook](https://github.com/primeqa/primeqa/blob/main/notebooks/tableqa/tableqa_inference.ipynb) shows how to test the pre-trained model available [here](https://huggingface.co/PrimeQA/tapas-based-tableqa-wikisql-lookup).
