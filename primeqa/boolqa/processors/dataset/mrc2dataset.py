import json
import argparse
import pandas as pd 
import sys
from primeqa.mrc.data_models.target_type import TargetType
from datasets import Dataset

def unpack_target_type(row):
    '''
    target_type_logits is an array of 5 numbers, with meanings determined by the 
    TargetType enum.  This routine unpacks them into separate keys for easier handling
    with pandas/datasets
    '''
    ttkey='target_type_logits'
    ttlogits=row[ttkey]
    for t in TargetType:
        key=f'{ttkey}_{str(t)}'
        row[key]=ttlogits[int(t)]
    #row.pop(ttkey)
    return row

def unpack_span_answer(row):
    '''
    span_answer is a dictionary with two items - split into separate items for easier
    handling with pandas/datasets
    '''
    row['span_answer_start_position'] = row['span_answer']['start_position']
    row['span_answer_end_position'] = row['span_answer']['end_position']
    row.pop('span_answer')
    return row

#----------------------------------------------------------------
def create_dataset_from_run_mrc_output(mrcfn: str, unpack: bool) -> Dataset:
    """Converts the output of run_mrc.py (eval_predictions.json, not eval_predictions_processed.json)
    into a Dataset for use with the classifiers

    Args:
        mrcfn (str): path to the eval_predictions.json file produced by run_mrc.py 

    Returns:
        Dataset : a dataset containing all of the fields in mrcfn as features, for the top-ranked answer
    """    
    with open(mrcfn) as mrcin:
        mrc=json.load(mrcin)
    return create_dataset_from_json_str(mrc, unpack)

def create_dataset_from_json_str(json_str: str, unpack: bool) -> Dataset:
    """Converts the output of run_mrc.py (eval_predictions.json, not eval_predictions_processed.json)
    into a Dataset for use with the classifiers

    Args:
        json_str (str): json encoding of the current state of eval_predictions

    Returns:
        Dataset : a dataset containing all of the fields in mrcfn as features, for the top-ranked answer
    """    

    # python dict class preserves order >3.6 - assuming it still works inside json
    def read_mrc_inner():
        for order, (key,vals) in enumerate(json_str.items()):
            for rank,val in enumerate(vals):
                if unpack:
                    val = unpack_target_type(val)
                    val = unpack_span_answer(val)
                val['order']=order
                val['rank']=rank
                yield val
    df=pd.DataFrame.from_records(read_mrc_inner())
    df0=df.query('rank==0') # TODO danger - assumes upstream results are correctly ordered
    ds=Dataset.from_pandas(df0, preserve_index=False)
    return ds
