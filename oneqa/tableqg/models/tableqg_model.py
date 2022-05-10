from transformers import AutoTokenizer,AutoModelWithLMHead

class TableQG():
    def __init__(self,model_path):
        self._model = AutoModelWithLMHead.from_pretrained(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    @property
    def model(self):
        return self._model
    @property
    def tokenizer(self):
        return self._tokenizer
