import argparse
import logging
from os import pread
import torch
import pandas as pd
import os
import ast

logger = logging.getLogger(__name__)

        
class WikiSQLPostprocessor:
    def __init__(self,tokenizer,args):
        self.tokenizer = tokenizer
        self.args = args

    def postprocess_prediction(self,eval_examples,eval_dataset, predictions):
        """Post process wikisql prediction using dev dataset and predictions

        Args:
            eval_examples (Examples): examples
            eval_dataset (Dataset): Development dataset object
            predictions (Tensor): prediction for each instance

        Returns:
            _type_: _description_
        """
        eval_preds = []
        gold_answers= []
        for i, data in enumerate(eval_dataset):
            data = {key: val.unsqueeze(0) for key, val in data.items()}
            answer_coordinates = self.tokenizer.convert_logits_to_predictions(data,torch.tensor(predictions[i,:]))
            gold_data = eval_dataset.data.iloc[i].to_dict()
            table_file = gold_data['table_file']
            gold_answer = ast.literal_eval(gold_data['answer_text'])
            predicted_answer = self.get_predicted_answer_text(table_file,answer_coordinates)
            eval_preds.append([str(i).lower() for i in predicted_answer])
            gold_answers.append([str(i).lower() for i in gold_answer])
        

        return eval_preds,gold_answers

    def get_predicted_answer_text(self,table_file,answer_coordinates):
        """The functions takes answer coordinates as input and predicts the answer texts

        Args:
            table_file (String): Name of the table file
            answer_coordinates (int,int): Row column index

        Returns:
            _type_: _description_
        """
        answers = []
        table = pd.read_csv(os.path.join(self.args.data_path_root,table_file),index_col=0).astype(str)
        for coordinates in answer_coordinates[0]:
            if len(coordinates) == 1:
            # only a single cell:
                answers.append(table.iat[coordinates[0]])
            else:
                cell_values = []
                for coordinate in coordinates:
                    cell_values.append(table.iat[coordinate])
                answers.append(", ".join(cell_values))
        return answers
    
