import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
from transformers import BertConfig, AutoModel
#from sentence_transformers import SentenceTransformer

from primeqa.ir.dense.colbert_top.colbert.utils.utils import torch_load_dnn
from primeqa.ir.dense.colbert_top.colbert.utils.utils import print_message


class HF_ColBERT_custom(BertPreTrainedModel):
    """
        Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.
        
        This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
    """
    _keys_to_ignore_on_load_unexpected = [r"cls"]

    def __init__(self, config, colbert_config):
        super().__init__(config)

        self.config = config
        self.dim = colbert_config.dim

        # get the config from the repo based on a name, e.g. 'sentence-transformers/all-MiniLM-L6-v2'
        bert_model_config = BertConfig.from_pretrained(config.name_or_path)
        # initalize the model structure
        self.bert = BertModel(bert_model_config)

        # get the values in the pretrained model
        pretrained_model = AutoModel.from_pretrained(config.name_or_path)

        # update the values based on pretrained_model
        from collections import OrderedDict
        import re

        state_dict = OrderedDict([(re.sub('0.auto_model.', '', k), v) for k, v in pretrained_model.state_dict().items()])
        self.bert.load_state_dict(state_dict)

        self.linear = nn.Linear(config.hidden_size, colbert_config.dim, bias=False)

        self.init_weights()

    @classmethod
    def from_pretrained(cls, name_or_path, colbert_config):
        if name_or_path.endswith('.dnn') or name_or_path.endswith('.model'):
            dnn = torch_load_dnn(name_or_path)

            base_default = 'bert-base-uncased'
            if (not dnn.get('arguments') or dnn.get('arguments').get('model')) and (not dnn.get('model_type')):
                print_message(f"[WARNING] Using default model type (base) {base_default}")
            base = dnn.get('arguments', {}).get('model', base_default) if dnn.get('arguments') else dnn.get('model_type', base_default)

            state_dict=dnn['model_state_dict']
            from collections import OrderedDict
            import re

            state_dict = OrderedDict([(re.sub(r'^model.', '', key), value) for key, value in state_dict.items()])

            obj = super().from_pretrained(base, state_dict=state_dict, colbert_config=colbert_config)
            obj.base = base

            return obj

        obj = super().from_pretrained(name_or_path, colbert_config=colbert_config)

        obj.base = name_or_path

        return obj

    def load_state_dict(self, name):
        assert name.endswith('dnn') or name.endswith('.model'), f"{name} is not valid colbert checkpoint ending with '.dnn' or '.model'"
        dnn = torch_load_dnn(name)
        state_dict = dnn['model_state_dict']

        import re
        from collections import OrderedDict

        state_dict = OrderedDict([(re.sub(r'^model.', '', key), value) for key, value in state_dict.items()])

        self.base = dnn['config']['_name_or_path']
        super().load_state_dict(state_dict)

    @staticmethod
    def raw_tokenizer_from_pretrained(name_or_path):
        if name_or_path.endswith('.dnn') or name_or_path.endswith('.model'):
            dnn = torch_load_dnn(name_or_path)
            base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')

            obj = AutoTokenizer.from_pretrained(base)
            obj.base = base

            return obj

        obj = AutoTokenizer.from_pretrained(name_or_path)
        obj.base = name_or_path

        return obj

    @property
    def bert(self):
        return self.model
