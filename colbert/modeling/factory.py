# bert imports
# from colbert.modeling.colbert import ColBERT
from colbert.modeling.hf_colbert import HF_ColBERT
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer



'''
# roberta imports
from colbert.modeling.colbert_roberta import ColBERT_Roberta
from colbert.modeling.tokenization import QueryTokenizerRoberta, DocTokenizerRoberta
'''

# xlmr imports
from colbert.modeling.hf_colbert_xlmr import HF_ColBERT_XLMR
from colbert.modeling.tokenization.doc_tokenization_xlmr import DocTokenizerXLMR
from colbert.modeling.tokenization.query_tokenization_xlmr import QueryTokenizerXLMR

'''
# mbert imports
from colbert.modeling.colbert_mbert import ColBERT_mbert
from colbert.modeling.tokenization import QueryTokenizerMBERT, DocTokenizerMBERT
'''
import os


# Based on model type to associate to a proper model and tokennizers(query, doc)
#----------------------------------------------------------------
def get_colbert_from_pretrained(name, colbert_config):
    # in V2, these come from
    # training::colbert = ColBERT(name=config.checkpoint, colbert_config=config)

    # currently, support bert and xlmr, ONLY and tinybert is hard wired.

    local_models_repository = colbert_config.local_models_repository
    model_type = name

    if model_type=='bert-base-uncased' or model_type=='bert-large-uncased':
        colbert = HF_ColBERT.from_pretrained(name, colbert_config)
    elif model_type == 'tinybert':
        if not local_models_repository:
            raise ValueError("Please specify the local repository for additional models.")
        #  colbert = ColBERT.from_pretrained(os.path.join(local_models_repository, 'tinybert/TinyBERT_General_4L_312D'), colbert_config)
        colbert = HF_ColBERT.from_pretrained(os.path.join(local_models_repository, 'tinybert/TinyBERT_General_4L_312D'), colbert_config)
        # e.g. from https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D/tree/main
    elif model_type=='roberta-base' or model_type=='roberta-large':
        colbert = ColBERT_Roberta.from_pretrained(model_type, colbert_config)
    elif model_type=='xlm-roberta-base' or model_type=='xlm-roberta-large':
        print (">>>>> using factory-made " + model_type)
        colbert = HF_ColBERT_XLMR.from_pretrained(name, colbert_config)
    elif model_type=='bert-base-multilingual-cased' or model_type=='bert-base-multilingual-uncased':
        colbert = ColBERT_mbert.from_pretrained(model_type, colbert_config)
    else:
        raise NotImplementedError()

    colbert.model_type=model_type
    return colbert

#----------------------------------------------------------------
def get_query_tokenizer(model_type, maxlen, attend_to_mask_tokens):
    # support and tested on bert and xmlr now, although leave other model type here
    if model_type=='bert-base-uncased' or model_type=='bert-large-uncased':
        return QueryTokenizer(maxlen,model_type, attend_to_mask_tokens)
    elif model_type=='tinybert':
        return QueryTokenizer(maxlen, 'bert-base-uncased',attend_to_mask_tokens)
    elif model_type=='roberta-base' or model_type=='roberta-large':
        return QueryTokenizerRoberta(maxlen, model_type)
    elif model_type=='xlm-roberta-base' or model_type=='xlm-roberta-large':
        return QueryTokenizerXLMR(maxlen, model_type)
    elif model_type=='bert-base-multilingual-cased':
        return QueryTokenizerMBERT(maxlen, model_type)
    elif model_type=='bert-base-multilingual-uncased':
        return QueryTokenizerMBERT(maxlen, model_type)
    else:
        raise NotImplementedError()

#----------------------------------------------------------------
def get_doc_tokenizer(model_type, maxlen):
    # support and tested on bert and xmlr now, although leave other model type here
    if model_type=='bert-base-uncased' or model_type=='bert-large-uncased':
        return DocTokenizer(maxlen, model_type)
    elif model_type=='tinybert':
        return DocTokenizer(maxlen, 'bert-base-uncased')
    elif model_type=='roberta-base' or model_type=='roberta-large':
        return DocTokenizerRoberta(maxlen, model_type)
    elif model_type=='xlm-roberta-base' or model_type=='xlm-roberta-large':
        return DocTokenizerXLMR(maxlen, model_type)
    elif model_type=='bert-base-multilingual-cased':
        return DocTokenizerMBERT(maxlen, model_type)
    elif model_type=='bert-base-multilingual-uncased':
        return DocTokenizerMBERT(maxlen, model_type)
    else:
        raise NotImplementedError()
