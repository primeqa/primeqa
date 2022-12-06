import logging
from pickle import NONE
from transformers import TapasConfig,TapasTokenizer, TapasForQuestionAnswering
from primeqa.mrc.run_mrc import ModelArguments, DataTrainingArguments, TrainingArguments
from primeqa.tableqa.tapas.run_tapas import  TableQAArguments
import pandas as pd
import numpy as np
import torch.utils.data
from transformers import (
    HfArgumentParser
)

from primeqa.tableqa.tapas.metrics.answer_accuracy import compute_denotation_accuracy
from primeqa.tableqa.tapas.postprocessor.wikisql import WikiSQLPostprocessor
from primeqa.tableqa.tapas.preprocessors.dataset import TableQADataset
from primeqa.tableqa.tapas.trainers.tableqa_trainer import TableQATrainer

from dataclasses import dataclass, field
from transformers import TapasConfig
from transformers import (
    DataCollator,
    HfArgumentParser,
    TrainingArguments,
    set_seed,default_data_collator,
)
import pandas as pd
from primeqa.tableqa.tapas.utils.data_collator import TapasCollator
from primeqa.tableqa.tapas.preprocessors.wikisql_preprocessor import load_data
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.WARNING)

class TapasModel():
    def __init__(self,path_to_config_json):
        """Tapas model class

        Args:
            path_to_config_json (str): Path to the configuration file in .json.
        """
        if path_to_config_json is not None:
            self._config_json = path_to_config_json
        else:
            self._config_json = "../../primeqa/tableqa/tapas/configs/tapas_config.json"

        # self._model = TapasForQuestionAnswering.from_pretrained(model_name_path)
        # self.config = config
        # self._tokenizer = TapasTokenizer.from_pretrained(model_name_path)


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
    
        
    def load_model_from_config(self,config_json) :
        print("loading from config at ",config_json)
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,TableQAArguments))
        model_args, data_args, training_args, tqa_args = parser.parse_json_file(json_file=os.path.abspath(config_json)) 

        config = TapasConfig(tqa_args)
        self._model = TapasForQuestionAnswering.from_pretrained(model_args.model_name_or_path)
        self._config = config
        self._tokenizer = TapasTokenizer.from_pretrained(model_args.model_name_or_path)
        
        return model_args, data_args, training_args, tqa_args
    
   
    def predict(self,data_dict,queries_list):
        """This function takes a table dictionary and a list of queries as input and returns the answer to the queries using the TableQA model.

        Args:
            data_dict (Dict): Table in dict format
            queries_list (List): List of queries

        Returns:
            Dict: Returns a dictionary of query and the predicted answer.
        """

        print("in predict for TapasModel with data: ",data_dict , " ,queries:", queries_list)
        self.load_model_from_config(self._config_json)

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

    def eval(self):
        model_args, data_args, training_args, tqa_args= self.load_model_from_config(self._config_json)
        post_obj = WikiSQLPostprocessor(self._tokenizer,tqa_args)
        
        if data_args.dataset_name=="wikisql":
            train_dataset,eval_dataset = load_data(tqa_args.data_path_root,self._tokenizer)
        else:
            tqadataset = TableQADataset(tqa_args.data_path_root,data_args.train_file,data_args.eval_file,self._tokenizer)
            train_dataset,eval_dataset= tqadataset.load_data()
        trainer = TableQATrainer(model=self._model,
                            args=training_args,
                            eval_dataset=eval_dataset,
                            tokenizer=self._tokenizer,
                            data_collator=TapasCollator(),
                            post_process_function= post_obj.postprocess_prediction,
                            compute_metrics=compute_denotation_accuracy  
                            )
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    def train(self):
        model_args, data_args, training_args, tqa_args= self.load_model_from_config(self._config_json)
        post_obj = WikiSQLPostprocessor(self._tokenizer,tqa_args)
        
        if data_args.dataset_name=="wikisql":
            train_dataset,eval_dataset = load_data(tqa_args.data_path_root,self._tokenizer)
        else:
            tqadataset = TableQADataset(tqa_args.data_path_root,data_args.train_file,data_args.eval_file,self._tokenizer)
            train_dataset,eval_dataset= tqadataset.load_data()
        
        #ToDo
        # if data_args.max_train_samples is not None:
        #     indices = np.random.permutation(len(train_dataset))[data_args.max_train_samples]
        #     train_dataset = Subset(train_dataset, indices)
        
        trainer = TableQATrainer(model=self._model,
                            args=training_args,
                            train_dataset=train_dataset,
                            tokenizer=self._tokenizer,
                            data_collator=TapasCollator(),
                            post_process_function= post_obj.postprocess_prediction,
                            compute_metrics=compute_denotation_accuracy  
                            )
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()



                                                                                                    
    
    


    




