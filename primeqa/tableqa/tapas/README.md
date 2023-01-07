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

# Table QA using TAPAS

Before continuing below make sure you have PrimeQA [installed](https://primeqa.github.io/primeqa/installation.html).

PrimeQA also supports answering questions over tables through run_mrc.

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

One can also run TAPAS without run_mrc using the TapasReader class defined in tapas_component class [here](./tapas_component.py).

Our python [notebook](https://github.com/primeqa/primeqa/blob/tapas_v2/notebooks/tableqa/tapas_inference.ipynb) shows how to use the TapasReader with a [config file](./configs/tapas_config.json) for inference using the pre-trained model available [here](https://huggingface.co/PrimeQA/tapas-based-tableqa-wikisql-lookup). TapasReader also supports training with a config file as shown in this [notebook](https://github.com/primeqa/primeqa/blob/tapas_v2/notebooks/tableqa/tapas_inference.ipynb)
