import os
import torch

from transformers import AutoTokenizer

from primeqa.ir.dense.colbert_top.colbert.utils.utils import torch_load_dnn
from primeqa.ir.dense.colbert_top.colbert.utils.utils import print_message
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig

from primeqa.ir.dense.colbert_top.colbert.modeling.factory import get_colbert_from_pretrained
from primeqa.ir.dense.colbert_top.colbert.modeling.factory import get_query_tokenizer, get_doc_tokenizer

class BaseColBERT(torch.nn.Module):
    """
    Shallow module that wraps the ColBERT parameters, custom configuration, and underlying tokenizer.
    This class provides direct instantiation and saving of the model/colbert_config/tokenizer package.

    Like HF, evaluation mode is the default.
    """

    def __init__(self, name, colbert_config=None):
        super().__init__()

        print_message(f"#>>>>> at BaseColBERT name (model type) : {name}")

        self.name = name
        self.colbert_config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint(name), colbert_config)
        # self.colbert_config = colbert_config
        # checkpoint_config = ColBERTConfig.load_from_checkpoint(name)
        # self.colbert_config.model_type = checkpoint_config.model_type

        self.model = get_colbert_from_pretrained(name, colbert_config=self.colbert_config)

        self.raw_tokenizer = AutoTokenizer.from_pretrained(self.model.base)
        # self.raw_tokenizer = None
        # TEMP fix
        # self.raw_tokenizer = get_doc_tokenizer(colbert_config.model_type, colbert_config.doc_maxlen)

        self.eval()

    @property
    def device(self):
        return self.model.device

    @property
    def bert(self):
        return self.model.bert

    @property
    def linear(self):
        return self.model.linear
    
    @property
    def score_scaler(self):
        return self.model.score_scaler

    def save(self, path):
        assert not path.endswith('.dnn'), f"{path}: We reserve *.dnn names for the deprecated checkpoint format."

        self.model.save_pretrained(path)
        self.raw_tokenizer.save_pretrained(path)

        self.colbert_config.save_for_checkpoint(path)


