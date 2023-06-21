from transformers import AutoConfig, PretrainedConfig

# bert imports
# from colbert.modeling.colbert import ColBERT
from primeqa.ir.dense.colbert_top.colbert.modeling.hf_colbert import HF_ColBERT
from primeqa.ir.dense.colbert_top.colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from primeqa.ir.dense.colbert_top.colbert.utils.utils import print_message

# xlmr imports
from primeqa.ir.dense.colbert_top.colbert.modeling.hf_colbert_xlmr import HF_ColBERT_XLMR
from primeqa.ir.dense.colbert_top.colbert.modeling.tokenization.doc_tokenization_xlmr import DocTokenizerXLMR
from primeqa.ir.dense.colbert_top.colbert.modeling.tokenization.query_tokenization_xlmr import QueryTokenizerXLMR

# Roberta imports
from primeqa.ir.dense.colbert_top.colbert.modeling.hf_colbert_roberta import HF_ColBERT_Roberta
from primeqa.ir.dense.colbert_top.colbert.modeling.tokenization.doc_tokenization_roberta import DocTokenizerRoberta
from primeqa.ir.dense.colbert_top.colbert.modeling.tokenization.query_tokenization_roberta import QueryTokenizerRoberta

import re
import os
import json
from collections import OrderedDict
from primeqa.ir.dense.colbert_top.colbert.utils.utils import torch_load_dnn


# Based on model type to associate to a proper model and tokennizers(query, doc)
#----------------------------------------------------------------
def get_colbert_from_pretrained(name, colbert_config):
    # in V2, these come from
    # training::colbert = ColBERT(name=config.checkpoint, colbert_config=config)

    # currently, we support bert, xlmr and roberta model types ONLY.

    print("printing inside the right branch")
    if name.endswith('.dnn') or name.endswith('.model'):
        dnn_checkpoint = torch_load_dnn(name)
        config = dnn_checkpoint.get('config', None)
        if config:
            delattr(PretrainedConfig, 'model_type')
            config = PretrainedConfig.from_dict(config)
            if not hasattr(config, 'hidden_size'):
                config.hidden_size = config.d_model
        checkpoint_model_type = dnn_checkpoint['model_type']
    else:
        checkpoint_config = AutoConfig.from_pretrained(name)
        checkpoint_model_type = checkpoint_config.model_type
        config = None

    assert checkpoint_model_type == colbert_config.model_type or \
            checkpoint_model_type.startswith(colbert_config.model_type), \
            f"Passed Model type {colbert_config.model_type} does \
            not match checkpoint Model type {checkpoint_model_type}"

    model_type = colbert_config.model_type
    print_message(f"factory model type: {model_type}")

    if model_type == 'bert':
        if config:
            colbert = HF_ColBERT(config, colbert_config)
            colbert.load_state_dict(name)
        else:
            colbert = HF_ColBERT.from_pretrained(name, colbert_config)
    elif model_type == 'xlm-roberta':
        if config:
            colbert = HF_ColBERT_XLMR(config, colbert_config)
            colbert.load_state_dict(name)
        else:
            colbert = HF_ColBERT_XLMR.from_pretrained(name, colbert_config)
    elif model_type == 'roberta':
        if config:
            colbert = HF_ColBERT_Roberta(config, colbert_config)
            colbert.load_state_dict(name)
        else:
            colbert = HF_ColBERT_Roberta.from_pretrained(name, colbert_config)
    else:
        raise NotImplementedError(f"Model type: {model_type} is not supported.")

    return colbert

#----------------------------------------------------------------
def get_query_tokenizer(name, colbert_config):
    model_type = colbert_config.model_type
    maxlen = colbert_config.query_maxlen
    attend_to_mask_tokens = colbert_config.attend_to_mask_tokens

    print_message(f"factory model type: {model_type}")

    if model_type == 'bert':
        return QueryTokenizer(maxlen, name, attend_to_mask_tokens)
    elif model_type == 'xlm-roberta':
        return QueryTokenizerXLMR(maxlen, name)
    elif model_type == 'roberta':
        return QueryTokenizerRoberta(maxlen, name)
    else:
        raise NotImplementedError(f"Model type: {model_type} is not supported.")

#----------------------------------------------------------------
def get_doc_tokenizer(name, colbert_config, is_teacher=False):
    model_type = colbert_config.model_type
    maxlen = colbert_config.teacher_doc_maxlen if is_teacher else colbert_config.doc_maxlen

    print_message(f"factory model type: {model_type}")

    if model_type == 'bert':
        return DocTokenizer(maxlen, name)
    elif model_type == 'xlm-roberta':
        return DocTokenizerXLMR(maxlen, name)
    elif model_type == 'roberta':
        return DocTokenizerRoberta(maxlen, name)
    else:
        raise NotImplementedError(f"Model type: {model_type} is not supported.")
