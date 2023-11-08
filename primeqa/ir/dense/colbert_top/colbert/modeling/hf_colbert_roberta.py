import re
from collections import OrderedDict

import torch.nn as nn

from transformers import AutoTokenizer, RobertaModel

from primeqa.ir.dense.colbert_top.colbert.utils.utils import torch_load_dnn
from primeqa.ir.dense.colbert_top.colbert.utils.utils import print_message


class HF_ColBERT_Roberta(RobertaModel):
    """
        Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.
        
        This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
    """

    def __init__(self, config, colbert_config):
        super().__init__(config)

        self.dim = colbert_config.dim
        self.encoder = None
        self.embeddings = None
        self.pooler = None

        self.config = config
        self.roberta = RobertaModel(config)
        self.linear = nn.Linear(config.hidden_size, colbert_config.dim, bias=False)

        self.init_weights()

    @classmethod
    def from_pretrained(cls, name_or_path, colbert_config):
        if name_or_path.endswith('.dnn') or name_or_path.endswith('.model'):
            dnn = torch_load_dnn(name_or_path)
            state_dict = dnn['model_state_dict']

            base_default = 'roberta-base'
            if (not dnn.get('arguments') or dnn.get('arguments').get('model')) and (not dnn.get('model_type')):
                print_message(f"[WARNING] Using default model type (base) {base_default}")
            base = dnn.get('arguments', {}).get('model', base_default) if dnn.get('arguments') else dnn.get('model_type', base_default)

            # for reading V2
            state_dict = OrderedDict([(re.sub(r'^model.', '', key), value) for key, value in state_dict.items() if 'bert.' not in key])

            obj = super().from_pretrained(base, state_dict=state_dict, colbert_config=colbert_config)

            return obj

        obj = super().from_pretrained(name_or_path, colbert_config=colbert_config)  # <<<< HERE

        return obj

    def load_state_dict(self, name):
        assert name.endswith('dnn') or name.endswith('.model'), f"{name} is not valid colbert checkpoint ending with '.dnn' or '.model'"
        dnn = torch_load_dnn(name)
        state_dict = dnn['model_state_dict']

        state_dict = OrderedDict([(re.sub(r'^model.', '', key), value) for key, value in state_dict.items() if 'bert.' not in key])

        super().load_state_dict(state_dict)

    @staticmethod
    def raw_tokenizer_from_pretrained(name_or_path):
        if name_or_path.endswith('.dnn') or name_or_path.endswith('.model'):
            dnn = torch_load_dnn(name_or_path)
            base = dnn.get('config', {}).get('_name_or_path', 'roberta-base')

            obj = AutoTokenizer.from_pretrained(base)

            return obj

        obj = AutoTokenizer.from_pretrained(name_or_path)

        return obj

    @property
    def bert(self):
        return self.roberta
