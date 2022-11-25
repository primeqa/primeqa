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


  #---------------------      

class RowClassifierSimple(nn.Module):
    def __init__(self):
        super(RowClassifierSimple, self).__init__()
        self.hidden_dim = 768
        self.num_labels = 2
        self.question_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.projection = nn.Sequential(nn.Linear( self.hidden_dim, self.hidden_dim), 
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_dim, self.num_labels))
   
    def forward(self,q_r_input,labels=None):
       #TODO: Three models in three separate GPUs 
        question_representation = self.question_encoder(**q_r_input).pooler_output
        logits = self.projection(question_representation)
        return logits


class RowClassifierQRSLongformer(nn.Module):
    def __init__(self):
        super(RowClassifierQRSLongformer, self).__init__()
        self.hidden_dim = 768
        self.num_labels = 2
        self.encoder = LongformerModel.from_pretrained('allenai/longformer-base-4096',gradient_checkpointing=True)
        self.projection = nn.Sequential(nn.Linear( self.hidden_dim, self.hidden_dim), 
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_dim, self.num_labels))
    def forward(self,q_r_input,labels=None):
        encoded = self.encoder(**q_r_input).pooler_output
        logits = self.projection(encoded)
        return logits
    
class RowClassifier(nn.Module):
    def __init__(self):
        super(RowClassifier, self).__init__()
        self.hidden_dim = 768
        self.num_labels = 2
        #self.proj_dim = 384
        self.question_encoder = RobertaModel.from_pretrained('roberta-base')
        #self.row_encoder = RobertaModel.from_pretrained('roberta-base')
        #self.passage_encoder = RobertaModel.from_pretrained('roberta-base')
        #with torch.no_grad():
        #self.passage_encoder = LongformerModel.from_pretrained('allenai/longformer-base-4096',gradient_checkpointing=True)
        #self.row_projection = nn.Linear(self.hidden_dim,self.proj_dim)
        #self.passage_projection = nn.Linear(self.hidden_dim,self.proj_dim)

        self.projection = nn.Sequential(nn.Linear( self.hidden_dim*4, self.hidden_dim), 
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_dim, self.num_labels))
   
    def forward(self,question_inputs,row_inputs,passage_inputs,labels=None):
       #TODO: Three models in three separate GPUs 
        concat_q_r = torch.cat(question_inputs,row_inputs)
        question_representation = self.question_encoder(**concat_q_r).pooler_output
        #row_representation = self.row_encoder(**row_inputs).pooler_output
        #q_dot_r = torch.mul(question_representation,row_representation)
    
        #passage_representation = self.passage_encoder(**passage_inputs).pooler_output
        #q_dot_p = torch.mul(question_representation,passage_representation)

        #print(passage_representation.shape)
        logits = self.projection(question_representation)
        #logits = self.projection(torch.cat([row_representation,passage_representation,q_dot_r,q_dot_p],-1)).squeeze()
        #probs = torch.softmax(logits, 0)
        #print(probs)
        return logits

