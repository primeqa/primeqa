# Table Question Answering(TableQA) model for question answering over tables

The primary script is [run_tableqa.py](./run_tableqa.py).  This runs a tapas based tableQA pipeline.
Before continuing below make sure you have PrimeQA [installed](../../README.md#Installation).

## Supported Datasets

Currently supported dataset
- WikiSQL
- SQA
- User's Custom Data

## Example Usage

An example usage for training and evaluation of TableQA model on default WikiSQL dataset :
```shell
python examples/tableqa/run_tableqa.py --do_train \
        --model_name_or_path "tapas-base" --do_eval \
        --dataset_name "wikisql" \
        --data_path_root "data/wikisql" \
        --output_dir <output_dir_path_to_save_the_model> \
        --learning_rate 4e-4
```
To train the model on user's data:

```shell
python examples/tableqa/run_tableqa.py --do_train \
        --model_name_or_path "tapas-base" --do_eval \
        --dataset_name "wikisql" \
        --data_path_root "data/my_root_dir" \
        --train_data_path "data/my_root_dir/train.tsv" \
        --dev_data_path "data/my_root_dir/dev.tsv"
        --output_dir <output_dir_path_to_save_the_model> \
        --learning_rate 4e-4

```

The format of dataset required for training and evaluation is:

`Question_id\tquestion\ttable_path\tanswer_coordinates\tanswer_text`    

The tables in csv format should be placed under `data_path_root/tables/`. The tables should have first row as column headers.


Refer to [notebooks](../notebooks/tableqa/) for knowing about how to test the pre-trained model available [here](https://huggingface.co/PrimeQA/tapas-based-tableqa-wikisql-lookup).



