from transformers import AutoTokenizer,AutoConfig, AutoModel

class TableQG():
    def __init__(self,model_path):
        super.__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    



        


