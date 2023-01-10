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

PrimeQA also supports answering questions over tables using TAPAS

For training and evaluation of a TAPAS based Table Question Answering model on wikisql dataset run the following script:
```shell
       python primeqa/tableqa/tapas/run_tapas.py \
       --model_name_or_path "PrimeQA/tapas-based-tableqa-wikisql-lookup" \
       --do_train \
       --do_eval \
       --dataset_name "wikisql" \
       --num_train_epochs 1.0 \
       --output_dir "models/wikisql/" \
       --data_path_root "data/wikisql/" \
       --num_aggregation_labels 4 \
       --use_answer_as_supervision true \
       --answer_loss_cutoff 0.664694 \
       --cell_selection_preference 0.207951 \
       --huber_loss_delta 0.121194 \
       --init_cell_selection_weights_to_zero true \
       --select_one_column true \
       --allow_empty_column_selection true \
       --temperature 0.0352513
'''
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
