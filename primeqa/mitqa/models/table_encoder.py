#from transformers import TapasTokenizer, TapasModel
from transformers import BertTokenizer, BertModel
from transformers import LongformerModel, LongformerTokenizer
from transformers import RobertaTokenizer,RobertaModel
from torch import nn
import torch


#from transformers import TapasTokenizer, TapasModel
from transformers import BertForSequenceClassification
from transformers import LongformerModel, LongformerTokenizer
from transformers import RobertaTokenizer,RobertaModel
from torch import nn
import torch
from transformers import TapasTokenizer, TapasModel


# Row Retriever model
class RowClassifierSC(nn.Module):
    def __init__(self,bert_model,state_dict=None):
        super(RowClassifierSC, self).__init__()
        self.num_labels = 2
        if state_dict:
            self.model = BertForSequenceClassification.from_pretrained(bert_model,state_dict=state_dict)
        else:
            self.model = BertForSequenceClassification.from_pretrained(bert_model)

    def forward(self,q_r_input,labels=None):
        outputs = self.model(**q_r_input,labels=labels)
        return outputs