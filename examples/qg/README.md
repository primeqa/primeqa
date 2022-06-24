# Table Question Generation (TableQG)

The primary script is [run_qg.py](./run_qg.py).  This runs a transformer-based sequence generation pipeline.
Before continuing below make sure you have OneQA [installed](../../README.md#Installation).

## Supported datasets
Currently supported datasets for training include:
- WikiSQL (QG for TableQA)
- SQuAD, SQuAD_v2 (QG for PassageQA i.e. MRC)

Inference can be done on any table in particular dict format. Check this [notebook](../../notebooks/tableqg/tableqg_inference.ipynb) for more information.

## Example Usage
An example for training the model on WikiSQL dataset:

```bash
python examples/tableqg/qg.py \
    --model_name_or_path t5-base \
    --modality table \
    --dataset_name wikisql \
    --do_train \
    --max_len 200 \
    --target_max_len 40 \
    --output_dir  models/qg/$DIR_NAME \
    --learning_rate 0.0001 \
    --num_train_epochs 4\
    --per_device_train_batch_size 32
```
Where the model_name_or_path can be any generator from huggingface ('t5', 'mt5', 'bart')
