## Description

### Custom Data

Users can train (fine-tune) and evaluate the MRC model on custom data by proving train_file and eval_file. 

Custom training and evaluation data may be provided in one of the formats supported by the available preprocessors such as SQUADPreprocessor or the BasePreProcessor. 

To run train and evaluation, use the run_mrc.py script as described [here](../../primeqa/mrc/README.md#example-usage) with the following additional parameters.

```shell
        --train_file "<path_to_train.json>" \
        --eval_file "<path_to_eval.json>" \
        --preprocessor <preprocessor> \
```

#### Custom Data in Huggingface SQUAD format
To run MRC on custom data in SQUAD format, specify the additional parameters as follows:

```shell
        --train_file "<path_to_train.json>" \
        --eval_file "<path_to_eval.json>" \
        --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
        --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
        --eval_metrics SQUAD 
```

Example custom training data is [here](./custom_data/examples_train_squad.json)
Example custom evaluation data is [here](./custom_data/examples_eval_squad.json)


#### Custom Data in PrimeQA Base format
To run MRC on custom data in PrimeQA Base format, use the following additional parameters:

```shell
       --train_file "<path_to_train.json>" \
       --eval_file "<path_to_eval.json>" \
       --preprocessor primeqa.mrc.processors.preprocessors.base.BasePreProcessor \
       --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
       --eval_metrics SQUAD 
```
TODO:
Example custom training data is [here](./custom_data/examples_train_base.json)
Example custom evaluation data is [here](./custom_data/examples_eval_base.json)
