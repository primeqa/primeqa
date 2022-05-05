from transformers import T5Tokenizer, T5ForConditionalGeneration


class TableQG(T5ForConditionalGeneration):
    def __init__(self,model_path):
        super.__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    



        


