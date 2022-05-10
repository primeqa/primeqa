from transformers import AutoTokenizer,AutoModel

class TableQG():
    def __init__(self,model_path):
        super.__init__()
        self._model = AutoModel.from_pretrained(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    @property
    def model(self):
        return self._model
    @property
    def tokenizer(self):
        return self._tokenizer

    



        


