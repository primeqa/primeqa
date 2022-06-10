import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
from transformers import AutoModel
from primeqa.ir.dense.colbert_top.colbert.utils.utils import torch_load_dnn
from transformers import PreTrainedModel
from transformers import XLMRobertaModel

class HF_ColBERT_XLMR(XLMRobertaModel):
#class HF_ColBERT_XLMR(PreTrainedModel):
    """
        Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.
        
        This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
    """

    def __init__(self, config, colbert_config):
        super().__init__(config)

        self.dim = colbert_config.dim
        # resolve conflict between bert and roberta
        # self.roberta = XLMRobertaModel(config)
        # self.bert = self.roberta
        self.encoder = None
        self.embeddings = None
        self.pooler = None

        self.roberta = XLMRobertaModel(config)
        self.bert = self.roberta
        #self.bert = XLMRobertaModel(config)

        self.linear = nn.Linear(config.hidden_size, colbert_config.dim, bias=False)

        # if colbert_config.relu:
        #     self.score_scaler = nn.Linear(1, 1)

        self.init_weights()

        # if colbert_config.relu:
        #     self.score_scaler.weight.data.fill_(1.0)
        #     self.score_scaler.bias.data.fill_(-8.0)

    @classmethod
    def from_pretrained(cls, name_or_path, colbert_config):
        if name_or_path.endswith('.dnn') or name_or_path.endswith('.model'):
            dnn = torch_load_dnn(name_or_path)
            base = dnn.get('arguments', {}).get('model', 'xlm-roberta-base')  # TODO: how about other lm-roberta-XXX?

            state_dict=dnn['model_state_dict']
            from collections import OrderedDict
            import re

            # for reading V1
            # state_dict = OrderedDict([(re.sub(r'^roberta.', 'bert.', key), value) for key, value in state_dict.items()])

            # for reading V2
            state_dict = OrderedDict([(re.sub(r'^model.', '', key), value) for key, value in state_dict.items()])

            obj = super().from_pretrained(base, state_dict=state_dict, colbert_config=colbert_config)
            #obj = super().from_pretrained(base, state_dict=dnn['model_state_dict'], colbert_config=colbert_config)
            obj.base = base

            return obj

        obj = super().from_pretrained(name_or_path, colbert_config=colbert_config)  # <<<< HERE

        obj.base = name_or_path

        return obj

    @staticmethod
    def raw_tokenizer_from_pretrained(name_or_path):
        if name_or_path.endswith('.dnn') or name_or_path.endswith('.model'):
            dnn = torch_load_dnn(name_or_path)
            base = dnn.get('arguments', {}).get('model',  'xlm-roberta-base')  # TODO: how about other lm-roberta-XXX?

            obj = AutoTokenizer.from_pretrained(base)
            obj.base = base

            return obj

        obj = AutoTokenizer.from_pretrained(name_or_path)
        obj.base = name_or_path

        return obj

"""
TODO: It's easy to write a class generator that takes "name_or_path" and loads AutoConfig to check the Architecture's
      name, finds that name's *PreTrainedModel and *Model in dir(transformers), and then basically repeats the above.

      It's easy for the BaseColBERT class to instantiate things from there.
"""

