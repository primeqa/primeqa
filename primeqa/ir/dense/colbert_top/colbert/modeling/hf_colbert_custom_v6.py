import torch.nn as nn
import torch
import os

from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
from transformers import AutoModel, AutoConfig
from primeqa.ir.dense.colbert_top.colbert.utils.utils import torch_load_dnn
from transformers import PreTrainedModel
from transformers import RobertaModel, RobertaConfig

class HF_ColBERT_custom_v6(RobertaModel):
#class HF_ColBERT_XLMR(PreTrainedModel):
    """
        Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.
        
        This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
    """

    def __init__(self, config, colbert_config):
        #config1 = RobertaConfig()
        # from nasa/model/nasa_demo_hf.ipynb
        '''config.vocab_size = 50261
        config.bos_token_id = 0
        config.pad_token_id = 1
        config.eos_token_id = 2
        config.pad_token_id = 0
        '''
        super().__init__(config)

        self.dim = colbert_config.dim
        self.encoder = None
        self.embeddings = None
        self.pooler = None

        self.roberta = RobertaModel(config)
        self.bert = self.roberta
        self.linear = nn.Linear(config.hidden_size, colbert_config.dim, bias=False)

        self.init_weights()


    @classmethod
    def from_pretrained(cls, name_or_path, colbert_config):
        if name_or_path.endswith('.dnn') or name_or_path.endswith('.model'):
            dnn = torch_load_dnn(name_or_path)
            base = dnn.get('arguments', {}).get('model', 'roberta-base')

            state_dict=dnn['model_state_dict']
            from collections import OrderedDict
            import re
            state_dict = OrderedDict([(re.sub(r'^model.', '', key), value) for key, value in state_dict.items()])

            config_dir = os.environ.get('CUSTOM_TOKENIZER_AND_MODEL_DIRECTORY')
            assert config_dir, f"Please specify the custom model tokenizer directory in CUSTOM_TOKENIZER_AND_MODEL_DIRECTORY environment variable"
            config = RobertaConfig.from_pretrained(config_dir)
            obj = super().from_pretrained(base, config=config, state_dict=state_dict, colbert_config=colbert_config)
            #obj = super().from_pretrained(base, state_dict=dnn['model_state_dict'], colbert_config=colbert_config)
            obj.base = base

            return obj
        elif os.path.isdir(name_or_path):
            obj = super().from_pretrained(name_or_path, colbert_config=colbert_config)  # <<<< HERE
            obj.base = name_or_path
            return obj
        else:
            state_dict_fn = os.environ.get('CUSTOM_MODEL_INITIAL_STATE_DICTIONARY')
            assert state_dict_fn, f"Please specify the custom model tokenizer directory in CUSTOM_TOKENIZER_AND_MODEL_DIRECTORY environment variable"
            state_dict = torch.load(state_dict_fn)
            base = 'roberta-base'

            config_dir = os.environ.get('CUSTOM_TOKENIZER_AND_MODEL_DIRECTORY')
            assert config_dir, f"Please specify the custom model tokenizer directory in CUSTOM_TOKENIZER_AND_MODEL_DIRECTORY environment variable"
            config = RobertaConfig.from_pretrained(config_dir)
            obj = super().from_pretrained(base, config=config, state_dict=state_dict, colbert_config=colbert_config)

            obj.base = base

            return obj

    @staticmethod
    def raw_tokenizer_from_pretrained(name_or_path):
        if os.path.isdir(name_or_path):
            tokenizer_dir = name_or_path
        else:
            tokenizer_dir = os.environ.get('CUSTOM_TOKENIZER_AND_MODEL_DIRECTORY')
            assert tokenizer_dir, f"Please specify the custom model tokenizer directory in CUSTOM_TOKENIZER_AND_MODEL_DIRECTORY environment variable"

        obj = AutoTokenizer.from_pretrained(tokenizer_dir)
        obj.base = 'roberta-base'

        return obj

"""
TODO: It's easy to write a class generator that takes "name_or_path" and loads AutoConfig to check the Architecture's
      name, finds that name's *PreTrainedModel and *Model in dir(transformers), and then basically repeats the above.

      It's easy for the BaseColBERT class to instantiate things from there.
"""

