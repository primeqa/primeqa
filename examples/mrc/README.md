## Custom Data

Users can run train (fine-tune) and evaluate the MRC model on benchmark dataset or custom data. 

To perform a fully functional train and inference procedure for the MRC components, the primary script to use is [run_mrc.py](../../primeqa/mrc/run_mrc.py).  This runs a transformer-based MRC pipeline.

Before continuing below make sure you have PrimeQA [installed](../../README.md#Installation).

Usage with benchmark datasets is described [here](../../primeqa/mrc/README.md#example-usage).
Usage with custom data is described [here](./custom_data/).

### Custom Data
Users can train (fine-tune) and evaluate the MRC model on custom data by providing a train_file and eval_file. 

Custom training and evaluation data may be provided in one of the formats supported by the available preprocessors such as SQUADPreprocessor or the BasePreProcessor. 

To run train and evaluation, use the run_mrc.py script as described [here](../../primeqa/mrc/README.md#example-usage) with the following additional parameters. 

```shell
        --train_file "<path_to_train.json>" \
        --eval_file "<path_to_eval.json>" \
        --preprocessor <preprocessor> \
```

#### Custom Data in Huggingface SQUAD format

This is an example of data in SQUAD format. 

```shell
    {"id": "4", 
    "question": "who was president when idaho became a territory?", 
    "context": "Idaho became a territory when President Abraham Lincoln signed the Territorial Act on March 4, 1863. Idaho became the 43rd state of the United States on July 3, 1890 when President Benjamin Harrison signed the Act.", 
    "answers": {"text": ["Abraham Lincoln"], "answer_start": [40]}}
```

The examples must be provided in a file where each line is an example in JSON format.
Example files are [training data file](./custom_data/examples_train_squad.json) and [evaluation data](./custom_data/examples_eval_squad.json).

A sample [run script](run_mrc.sh) has been provided to get started using data in SQuAD format. 

To run MRC on custom data in SQUAD format, specify the additional parameters as follows:

```shell
        --train_file "<path_to_train.json>" \
        --eval_file "<path_to_eval.json>" \
        --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
        --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
        --eval_metrics SQUAD 
```

#### Custom Data in PrimeQA Base format

This is an example in the PrimeQA base format:
```shell
    {"question":"What is the oldest state highway in Georgia?",
    "language":"english",
    "target":{"end_positions":[-1,-1,-1],
        "passage_indices":[-1,-1,-1],
        "start_positions":[-1,-1,-1],
        "yes_no_answer":["NONE","NONE","NONE"]},
    "context":["\n\nIn the United States, each state maintains its own system of state highways.[lower-alpha 1] This is a list of the longest state highways in each state. As of 2007, the longest state highway in the nation is Montana Highway 200, which is 706.624 miles (1,137.201km) long. The shortest of the longest state highways is District of Columbia Route 295, which is only 4.29 miles (6.90km) long.\nList of highways\n\nNotes\n\nSee also\nU.S. Roadsportal\n\n\n\n Media related to State highways in the United States at Wikimedia Commons\n\n"],
    "example_id":"a00d476e-22c4-4fcd-9e89-f1fee10bd110"}
```
The examples must be provided in a file where each line is an example in JSON format.
Example files are: [training data file ](./custom_data/examples_train_base.json), and [evaluation data file](./custom_data/examples_eval_base.json)

To run MRC on custom data in PrimeQA Base format, use the following additional parameters:

```shell
       --train_file "<path_to_train.json>" \
       --eval_file "<path_to_eval.json>" \
       --preprocessor primeqa.mrc.processors.preprocessors.base.BasePreProcessor \
       --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
       --eval_metrics SQUAD 
```

### Custom Data in own Custom format

Alternatively, a user can create their own pre-processor to be used with their custom data format. The preprocessor must inherit from the [BasePreProcessor](https://github.com/primeqa/primeqa/blob/mrc-user-data/primeqa/mrc/processors/preprocessors/base.py). The existing pre-processors can be used as a template to get started.
