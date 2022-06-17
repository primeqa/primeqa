# bert imports
# from colbert.modeling.colbert import ColBERT
from primeqa.ir.dense.colbert_top.colbert.modeling.hf_colbert import HF_ColBERT
from primeqa.ir.dense.colbert_top.colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from primeqa.ir.dense.colbert_top.colbert.utils.utils import print_message


# xlmr imports
from primeqa.ir.dense.colbert_top.colbert.modeling.hf_colbert_xlmr import HF_ColBERT_XLMR
from primeqa.ir.dense.colbert_top.colbert.modeling.tokenization.doc_tokenization_xlmr import DocTokenizerXLMR
from primeqa.ir.dense.colbert_top.colbert.modeling.tokenization.query_tokenization_xlmr import QueryTokenizerXLMR

import os
import json
from primeqa.ir.dense.colbert_top.colbert.utils.utils import torch_load_dnn

# Based on model type to associate to a proper model and tokennizers(query, doc)
#----------------------------------------------------------------
def get_colbert_from_pretrained(name, colbert_config):
    # in V2, these come from
    # training::colbert = ColBERT(name=config.checkpoint, colbert_config=config)

    # currently, support bert and xlmr, ONLY and tinybert is hard wired.

    local_models_repository = colbert_config.local_models_repository
    model_type = name

    if colbert_config.model_type is not None:
        model_type = colbert_config.model_type

    # if it is a directory, load json file to get the model type,or  if it is a dnn file
    if os.path.isdir(name):
        json_file= name + '/config.json'
        print_message(f"json file (get_colbert_from_pretrained): {json_file}")
        with open(json_file) as file:
            data = json.load(file)
        assert model_type == data["_name_or_path"], f"model type in {name} not matching"
        # model_type = data["_name_or_path"]
    elif name.endswith('.dnn') or name.endswith('.model'):
        dnn_checkpoint = torch_load_dnn(name)
        assert model_type == dnn_checkpoint['model_type'], f"model type in {name} not matching"
        # model_type = dnn_checkpoint['model_type']

    print_message(f"factory model type: {model_type}")

    if model_type=='bert-base-uncased' or model_type=='bert-large-uncased':
        colbert = HF_ColBERT.from_pretrained(name, colbert_config)
    elif model_type == 'tinybert':
        if not local_models_repository:
            raise ValueError("Please specify the local repository for additional models.")
        #  hard wired for local Tinybert model
        colbert = HF_ColBERT.from_pretrained(os.path.join(local_models_repository, 'tinybert/TinyBERT_General_4L_312D'), colbert_config)
        # e.g. from https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D/tree/main
    elif model_type=='xlm-roberta-base' or model_type=='xlm-roberta-large':
        colbert = HF_ColBERT_XLMR.from_pretrained(name, colbert_config)
    else:
        raise NotImplementedError

    colbert.model_type=model_type
    return colbert

#----------------------------------------------------------------
def get_query_tokenizer(model_type, maxlen, attend_to_mask_tokens):


    # if it is a directory, load json file to get the model type
    if os.path.isdir(model_type):
        json_file = model_type + '/config.json'
        print_message(f"json file (get_query_tokenizer): {json_file}")
        with open(json_file) as file:
            data = json.load(file)
        model_type = data["_name_or_path"]

    print_message(f"get query model type: {model_type}")

    if model_type=='bert-base-uncased' or model_type=='bert-large-uncased':
        return QueryTokenizer(maxlen,model_type, attend_to_mask_tokens)
    elif model_type=='tinybert':
        return QueryTokenizer(maxlen, 'bert-base-uncased',attend_to_mask_tokens)
    elif model_type=='xlm-roberta-base' or model_type=='xlm-roberta-large':
        return QueryTokenizerXLMR(maxlen, model_type)
    else:
        raise NotImplementedError

#----------------------------------------------------------------
def get_doc_tokenizer(model_type, maxlen):


    # if it is a directory, load json file to get the model type
    if os.path.isdir(model_type):
        json_file = model_type + '/config.json'
        print_message(f"json file (get_doc_tokenizer): {json_file}")
        with open(json_file) as file:
            data = json.load(file)
        model_type = data["_name_or_path"]


    print_message(f"get doc model type: {model_type}")

    if model_type=='bert-base-uncased' or model_type=='bert-large-uncased':
        return DocTokenizer(maxlen, model_type)
    elif model_type=='tinybert':
        return DocTokenizer(maxlen, 'bert-base-uncased')
    elif model_type=='xlm-roberta-base' or model_type=='xlm-roberta-large':
        return DocTokenizerXLMR(maxlen, model_type)
    else:
        raise NotImplementedError
