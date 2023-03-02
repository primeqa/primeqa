## Custom MRC Data

This README describes how to train (fine-tune) and evaluate [run_mrc.py](../../primeqa/mrc/run_mrc.py) on custom data. 
Usage with benchmark datasets is described [here](../../primeqa/mrc/README.md#example-usage).

### Prerequisites

Before continuing below make sure you have PrimeQA [installed](../../README.md#Installation).

### Getting Started
Users can train (fine-tune) and evaluate the MRC model on custom data by providing a train_file and eval_file. 

Custom training and evaluation data may be provided in one of the formats supported by any of the available preprocessors such as SQUADPreprocessor or the BasePreProcessor. Sample run scripts for [training and inference](run_mrc_train_eval.sh) and [inference only](run_mrc_eval.sh) have been provided to get started using data in SQuAD format. The [SQuADPreprocessor format](./README.md#Custom-Data-in-Huggingface-SQUAD-format) only supports the basic setup where the answer is a span in the provided passage. For more complex datasets (e.g. boolean question, multiple passages) use the [BasePreprocessor format](./README.md#Custom-Data-in-PrimeQA-Base-format). Finally, if the existing preprocessors do not support your data you must make a [custom preprocessor](./README.md#Custom-Data-in-own-custom-format).  

### Fine-Tuning Model

The starting point of fine-tuning on custom data can use an already trained model available in our [model hub](https://huggingface.co/PrimeQA). For example, this [model](https://huggingface.co/PrimeQA/squad-v1-roberta-large) trained on [SQuAD 1.1](https://aclanthology.org/D16-1264/). On the other hand, one can completely start fresh with a model initialized with a large pre-trained language model e.g. [RoBERTa](https://huggingface.co/roberta-large/) and fine-tune on their custom data. Note: typically, starting with an already fine-tuned model on SQuAD 1.1 is better than starting fresh on your own custom data. The model used for fine-tuning on your dataset can be set as follows:

#### [PrimeQA SQuAD 1.1](https://huggingface.co/PrimeQA/squad-v1-roberta-large)
```shell
--model_name_or_path PrimeQA/squad-v1-roberta-large
```

#### [RoBERTa](https://huggingface.co/roberta-large/)
```shell
--model_name_or_path roberta-large
```

### Custom Data in Huggingface SQUAD format

The custom data must be provided in a file where each line is an example in JSON format.
Example files are [training data](./custom_data/examples_train_squad.jsonl) and [evaluation data](./custom_data/examples_eval_squad.jsonl).
This is an example of data in SQUAD format from [HF datasets](https://huggingface.co/datasets/squad/viewer/plain_text/train). 

```shell
    {"id": "4", 
    "question": "who was president when idaho became a territory?", 
    "context": "Idaho became a territory when President Abraham Lincoln signed the Territorial Act on March 4, 1863. Idaho became the 43rd state of the United States on July 3, 1890 when President Benjamin Harrison signed the Act.", 
    "answers": {"text": ["Abraham Lincoln"], "answer_start": [40]}}
```

To run MRC on custom data in SQUAD format, specify the parameters as follows (A sample run script has been provided to get started using data in SQuAD format (You can use the [training and inference](run_mrc_train_eval.sh) and [inference only](run_mrc_eval.sh) run scripts to get started):

```shell
        --train_file "<path_to_train.jsonl>" \
        --eval_file "<path_to_eval.jsonl>" \
        --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
        --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
        --eval_metrics SQUAD 
```

### Custom Data in PrimeQA Base format

The custom data must be provided in a file where each line is an example in JSON format.
Example files are: [training data](./custom_data/examples_train_base.jsonl), and [evaluation data](./custom_data/examples_eval_base.jsonl).
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

To run MRC on custom data in PrimeQA Base format, use the following parameters (You can modify the preprocessor in [training and inference](run_mrc_train_eval.sh) and [inference only](run_mrc_eval.sh) to get started):

```shell
       --train_file "<path_to_train.jsonl>" \
       --eval_file "<path_to_eval.jsonl>" \
       --preprocessor primeqa.mrc.processors.preprocessors.base.BasePreProcessor \
       --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
       --eval_metrics SQUAD 
```

### Custom Data in own Custom format

Alternatively, a user can create their own pre-processor to be used with their custom data format. The preprocessor must inherit from the [BasePreProcessor](https://github.com/primeqa/primeqa/blob/mrc-user-data/primeqa/mrc/processors/preprocessors/base.py). The existing pre-processors can be used as a template to get started.


### Finetuning using feedback data

To finetune an extractive reader model using feedback data collected using the [PrimeQA application](https://github.com/primeqa/primeqa-orchestrator), first download the feedback data as described and convert the feedback json file into a jsonl file using the following command:

```
python convert_jsonarray_to_jsonl.py --input_file <path-to-feedback-json-file> --output_file <path-to-jsonl-file>
```

The jsonl file can be passed in as training data as described [Custom Data in Huggingface SQUAD format](###-Custom-Data-in-Huggingface-SQUAD-format)
