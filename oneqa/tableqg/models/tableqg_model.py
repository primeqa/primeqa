from transformers import AutoTokenizer,AutoModelWithLMHead

class TableQG():
    def __init__(self,model_path):
        """ Table Question Generation Model gets initialized based on either pre-trained model path or
        the model name. One example could be 't5-base'.

        Args:
            model_path (String): Either Name of the model or the path to the pre-trained model
        """        
        self._model = AutoModelWithLMHead.from_pretrained(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
    
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
