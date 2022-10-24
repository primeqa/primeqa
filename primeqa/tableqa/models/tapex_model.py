from pickle import NONE
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import pandas as pd

class TapexModel():
    def __init__(self,model_name_path):
        """TableQA model class

        Args:
            model_name_path (str): Path to the pre-trained model.
            config (_type_, optional): _description_. Defaults to None.
        """
        self._tokenizer  = AutoTokenizer.from_pretrained(model_name_path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name_path)


    @property
    def model(self):
        """ Propery of TableQA model.
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
        

        

    def predict(self,data_dict,queries_list):
        """This function takes a table dictionary and a list of queries as input and returns the answer to the queries using the TableQA model.

        Args:
            data_dict (Dict): Table in dict format
            queries_list (List): List of queries

        Returns:
            Dict: Returns a dictionary of query and the predicted answer.
        """
        table = pd.DataFrame.from_dict(data_dict)
        inputs = self._tokenizer(table, queries_list, padding='max_length', return_tensors="pt")
        print(queries_list)
        outputs = self._model.generate(**inputs)
        answers = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(answers)
        

        


        query_answer_dict = {}
        for query, answer in zip(queries_list, answers):
            query_answer_dict[query] = answer
        return query_answer_dict
                                                                                                    
    
    


    




