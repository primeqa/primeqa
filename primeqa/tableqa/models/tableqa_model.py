from transformers import TapasConfig,TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

class TableQAModel():
    def __init__(self,model_name,config=None):
        self._model = TapasForQuestionAnswering.from_pretrained(model_name)
        self.config = config
        self._tokenizer = TapasTokenizer.from_pretrained(model_name)


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
        

    def predict_from_dict(self,data_dict,queries_list):
        table = pd.DataFrame.from_dict(data_dict)
        inputs = self._tokenizer(table=table, queries=queries_list, padding='max_length', return_tensors="pt")
        outputs = self._model(**inputs)
        print(outputs)
        predicted_answer_coordinates, predicted_aggregation_indices = self._tokenizer.convert_logits_to_predictions(inputs,
                                                                                                            outputs.logits.detach(),
                                                                                                            outputs.logits_aggregation.detach())
        id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3:"COUNT"}
        aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]
        answers = []
        for coordinates in predicted_answer_coordinates:
            if len(coordinates) == 1:
            # only a single cell:
                answers.append(table.iat[coordinates[0]])
            else:
                cell_values = []
                for coordinate in coordinates:
                    cell_values.append(table.iat[coordinate])
                answers.append(", ".join(cell_values))
        query_answer_dict = {}
        for query, answer, predicted_agg in zip(queries_list, answers, aggregation_predictions_string):
            print(query)
            
            if predicted_agg == "NONE":
                print("Predicted answer: " + answer)
            else:
                print("Predicted answer: " + predicted_agg + " > " + answer)
            query_answer_dict[query] = answer
        return query_answer_dict
                                                                                                    
    
    


    




