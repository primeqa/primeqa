import torch.nn as nn
import torch

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
        # resolve conflict between bert and roberta
        # self.roberta = XLMRobertaModel(config)
        # self.bert = self.roberta
        self.encoder = None
        self.embeddings = None
        self.pooler = None

        self.roberta = RobertaModel(config)
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
            base = dnn.get('arguments', {}).get('model', 'roberta-base')

            state_dict=dnn['model_state_dict']
            from collections import OrderedDict
            import re
            state_dict = OrderedDict([(re.sub(r'^model.', '', key), value) for key, value in state_dict.items()])

            config = RobertaConfig()
            # from nasa/model/nasa_demo_hf.ipynb
            # config.vocab_size = 50261
            # based on reading /dccstor/colbert-ir/franzm/nasa/model/v6/nasa-wiki-weighted-tokenizer-10-3-22/vocab.json
            config.vocab_size = 65536
            config.bos_token_id = 0
            config.pad_token_id = 1
            config.eos_token_id = 2
            config.pad_token_id = 0
            obj = super().from_pretrained(base, config=config, state_dict=state_dict, colbert_config=colbert_config)
            #obj = super().from_pretrained(base, state_dict=dnn['model_state_dict'], colbert_config=colbert_config)
            obj.base = base

            return obj

        # this is only used the first time a "bare" model state dictionary is loaded, subsequent fine-tuned models use the checkpoint
        state_dict_fn = '/dccstor/colbert-ir/franzm/nasa/model/v6/v6_nasawiki_nopad.pth'
        state_dict = torch.load(state_dict_fn)
        base = 'roberta-base'
        config = RobertaConfig()
        # was from nasa/model/nasa_demo_hf.ipynb
        # config.vocab_size = 50261
        # based on reading /dccstor/colbert-ir/franzm/nasa/model/v6/nasa-wiki-weighted-tokenizer-10-3-22/vocab.json
        config.vocab_size = 65536
        config.bos_token_id = 0
        config.pad_token_id = 1
        config.eos_token_id = 2
        config.pad_token_id = 0
        #config1 = AutoConfig.from_pretrained('/dccstor/bsiyer6/public/nasa/model')
        obj = super().from_pretrained(base, config=config, state_dict=state_dict, colbert_config=colbert_config)
        #hf_model.load_state_dict(state_dict, strict=False)

        obj.base = base

        return obj

    @staticmethod
    def raw_tokenizer_from_pretrained(name_or_path):
        '''if name_or_path.endswith('.dnn') or name_or_path.endswith('.model'):
            dnn = torch_load_dnn(name_or_path)
            base = dnn.get('arguments', {}).get('model',  'xlm-roberta-base')  # TODO: how about other lm-roberta-XXX?

            obj = AutoTokenizer.from_pretrained(base)
            obj.base = base

            return obj
        '''

        tokenizer_dir = '/dccstor/colbert-ir/franzm/nasa/model/v6/nasa-wiki-weighted-tokenizer-10-3-22'
        obj = AutoTokenizer.from_pretrained(tokenizer_dir)
        obj.base = 'roberta-base'

        return obj

"""
TODO: It's easy to write a class generator that takes "name_or_path" and loads AutoConfig to check the Architecture's
      name, finds that name's *PreTrainedModel and *Model in dir(transformers), and then basically repeats the above.

      It's easy for the BaseColBERT class to instantiate things from there.
"""

