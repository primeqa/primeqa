from pickle import NONE
from transformers import TapasConfig,TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

class TableQAModel():
    def __init__(self,model_name_path,config=None):
        """TableQA model class

        Args:
            model_name_path (str): Path to the pre-trained model.
            config (_type_, optional): _description_. Defaults to None.
        """
        self._model = TapasForQuestionAnswering.from_pretrained(model_name_path)
        self.config = config
        self._tokenizer = TapasTokenizer.from_pretrained(model_name_path)


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
        inputs = self._tokenizer(table=table, queries=queries_list, padding='max_length', return_tensors="pt")
        outputs = self._model(**inputs)
        predicted_answer_coordinates = self._tokenizer.convert_logits_to_predictions(inputs,
                                                                                    outputs.logits.detach(),
                                                                                    None)
        answers = []
        for coordinates in predicted_answer_coordinates[0]:
            if len(coordinates) == 1:
            # only a single cell:
                answers.append(table.iat[coordinates[0]])
            else:
                cell_values = []
                for coordinate in coordinates:
                    cell_values.append(table.iat[coordinate])
                answers.append(", ".join(cell_values))
        query_answer_dict = {}
        for query, answer in zip(queries_list, answers):
            query_answer_dict[query] = answer
        return query_answer_dict
                                                                                                    
    
    


    




