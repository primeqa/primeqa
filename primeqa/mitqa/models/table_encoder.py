from torch import nn
import torch
from transformers import RobertaTokenizer,RobertaModel
from transformers import BertForSequenceClassification



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
        """
        The forward function is the core of a PyTorch module. It defines what happens when you call an instantiated
        object of this class with one or more tensors as arguments. In this case, it takes in two tensors: the raw
        question and passage text, and returns a single output (the un-normalized logits for each class). The forward function
        is where most of your code will go.
        
        Args:
            self: Access the attributes and methods of the class
            q_r_input: Pass the question and the passage to the model
            labels: Calculate the loss function
        
        Returns:
            The output of the model
        """
        outputs = self.model(**q_r_input,labels=labels)
        return outputs