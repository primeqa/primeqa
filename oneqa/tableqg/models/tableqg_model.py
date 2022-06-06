from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from oneqa.tableqg.utils.constants import SQLSpecialTokens

class TableQG():
    def __init__(self,model_path):
        """ Table Question Generation Model gets initialized based on either pre-trained model path or
        the model name. One example could be 't5-base'.

        Args:
            model_path (String): Either Name of the model or the path to the pre-trained model
        """        
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        # adding special tokens to tokenizer which is needed to create SQL
        # expanding token embeddings in model
        
        sql_tokens_list = [SQLSpecialTokens.sep, SQLSpecialTokens.cond, SQLSpecialTokens.ans,
                        SQLSpecialTokens.header, SQLSpecialTokens.hsep]
        for sql_token in sql_tokens_list:
            if sql_token not in self._tokenizer.vocab: # add when special-tokens aren't already there
                self._tokenizer.add_tokens([sql_token])
        self._model.resize_token_embeddings(len(self._tokenizer.vocab))
    
    @property
    def model(self):
        """ Propery of TableQG model.

        Returns:
            Sequence to sequence model object (based on model name)
        """
        return self._model
    @property
    def tokenizer(self):
        """ Property of TableQG model.

        Returns:
            Tokenizer class object based on the model name/ path
        """
        return self._tokenizer
