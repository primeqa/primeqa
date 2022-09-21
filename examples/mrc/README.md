## Description

Users can run train (fine-tune) and evaluate the MRC model on benchmark dataset or custom data. 

To perform a fully functional train and inference procedure for the MRC components, the primary script to use is [run_mrc.py](../../primeqa/mrc/run_mrc.py).  This runs a transformer-based MRC pipeline.

Before continuing below make sure you have PrimeQA [installed](../../README.md#Installation).

Usage with benchmark datasets is described [here](../../primeqa/mrc/README.md#example-usage).
Usage with custom data is described [here](./custom_data/).

### Custom Data

Custom training and evaluation data may be provided in one of the formats supported by the available preprocessors such as SQUADPreprocessor or the BasePreProcessor. 

To run train and evaluation, use the run_mrc.py script as described [here](../../primeqa/mrc/README.md#example-usage) with the following additional parameters.

```shell
        --train_file "<path_to_train.json>" \
        --eval_file "<path_to_eval.json>" \
        --preprocessor <preprocessor> \
```

#### Custom Data in Huggingface SQUAD format

This is an example of data in SQUAD format. Exa

```shell
    {"id": "4", 
    "question": "who was president when idaho became a territory?", 
    "context": "Idaho became a territory when President Abraham Lincoln signed the Territorial Act on March 4, 1863. Idaho became the 43rd state of the United States on July 3, 1890 when President Benjamin Harrison signed the Act.", 
    "answers": {"text": ["Abraham Lincoln"], "answer_start": [40]}}
```

The examples must be provided in a file where each line is an example in JSON format.
Example custom training data file is [here](./custom_data/examples_train_squad.json)
Example custom evaluation data file is [here](./custom_data/examples_eval_squad.json)

To run MRC on custom data in SQUAD format, specify the additional parameters as follows:

```shell
        --train_file "<path_to_train.json>" \
        --eval_file "<path_to_eval.json>" \
        --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
        --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
        --eval_metrics SQUAD 
```

#### Custom Data in PrimeQA Base format

TODO:
Example: 



The examples must be provided in a file where each line is an example in JSON format.
Example custom training data is [here](./custom_data/examples_train_base.json)
Example custom evaluation data is [here](./custom_data/examples_eval_base.json)

To run MRC on custom data in PrimeQA Base format, use the following additional parameters:

```shell
       --train_file "<path_to_train.json>" \
       --eval_file "<path_to_eval.json>" \
       --preprocessor primeqa.mrc.processors.preprocessors.base.BasePreProcessor \
       --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
       --eval_metrics SQUAD 
```

