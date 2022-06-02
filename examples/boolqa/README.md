# TydiQA with support for Boolean questions

The TydiQA dataset can be decoded, with full support for both the boolean and short-answer questions in two different ways.
As a single command line argument in `run_mrc.py` to run all steps:

```shell
python examples/mrc/run_mrc.py --model_name_or_path ${TRAINING_OUTPUT_DIR} \
       --output_dir ${OUTPUT_DIR} --fp16 --do_eval \
       --do_boolean \
       --per_device_eval_batch_size 128 --overwrite_output_dir
```
or step-by-step.
The output of each individual step is analyzed in more detail [here](../../notebooks/boolqa/eval_predictions.ipynb)

## MRC
## EVC 
## SN
