from typing import List
from dataclasses import dataclass, field
import json
import numpy as np

from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    DataCollatorWithPadding, 
    AutoModelForSequenceClassification
)
from datasets import Dataset

from primeqa.pipelines.components.base import ReaderComponent
from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from primeqa.text_classification.processors.postprocessors.text_classifier import TextClassifierPostProcessor
from primeqa.text_classification.processors.preprocessors.text_classifier import TextClassifierPreProcessor
from primeqa.text_classification.trainers.nway import NWayTrainer


@dataclass
class BooleanQTCReader(ReaderComponent):
    """_summary_

    Args:
        model (str, optional): Model. Defaults to "PrimeQA/nq_tydi_sq1-reader-xlmr_large-20221110".
        use_fast (bool, optional): If set to "True", uses the fast version of the tokenizer. Defaults to True.
        stride (int, optional): Step size to move sliding window across context. Defaults to 128.
        max_seq_len (int, optional): Maximum length of question and context inputs to the model (in word pieces/bpes). Defaults to 512.
        n_best_size (int, optional): Maximum number of start/end logits to consider (max values). Defaults to 20.
        max_num_answers (int, optional): Maximum number of answers. Defaults to 5.
        max_answer_length (int, optional): Maximum answer length. Defaults to 32.
        scorer_type (str, optional): Scoring algorithm. Defaults to "weighted_sum_target_type_and_score_diff".
        min_score_threshold: (float, optional): Minimum score threshold. Defaults to None.

    Important:
        1. Each field has metadata property which can carry additional information for other downstream usages.
        2. Two special keys (api_support and exclude_from_hash) are defined in "metadata" property.
            a. api_support (bool, optional): If set to True, that parameter is exposed via service layer. Defaults to False.
            b. exclude_from_hash (bool,optional): If set to True, that parameter is not considered while building the hash representation for the object. Defaults to False.


    Returns:
        _type_: _description_
    """

    model: str = field(
        default="PrimeQA/tydi-tydi_boolean_question_classifier-xlmr_large-20221117",
        metadata={"name": "Model", "api_support": True},
    )
    use_fast: bool = field(
        default=True,
        metadata={
            "name": "Use the fast version of the tokenizer",
            "options": [True, False],
        },
    )
    # stride: int = field(
    #     default=128,
    #     metadata={
    #         "name": "Stride",
    #         "description": "Step size to move sliding window across context",
    #         "range": [8, 256, 8],
    #     },
    # )
    max_seq_len: int = field(
        default=512,
        metadata={
            "name": "Maximum sequence length",
            "description": "Maximum length of question and context inputs to the model (in word pieces/bpes)",
            "range": [32, 512, 8],
        },
    )
    n_best_size: int = field(
        default=20,
        metadata={
            "name": "N",
            "description": "Maximum number of start/end logits to consider (max values)",
            "range": [1, 50, 1],
        },
    )
    max_num_answers: int = field(
        default=3,
        metadata={
            "name": "Maximum number of answers",
            "range": [1, 5, 1],
            "api_support": True,
            "exclude_from_hash": True,
        },
    )
    max_answer_length: int = field(
        default=1000,
        metadata={
            "name": "Maximum answer length",
            "range": [2, 2000, 2],
            "api_support": True,
            "exclude_from_hash": True,
        },
    )
    id_key: str = field(
        default='example_id',
        metadata={
            "name": "unique identifier for examples",
            "api_support": False,
            "exclude_from_hash": True,                
        }
    )
    output_label_prefix: str = field(
        default='qtc',
        metadata={
            "name": "prefix for output labels",
            "api_support": False,
            "exclude_from_hash": False,       
        }
    )    
    sentence1_key: str = field(
        default='question',
        metadata={
            "name": "sentence1 key for preprocessor",
            "api_support": False,
            "exclude_from_hash": True,            
        },
    )
    sentence2_key: str = field(
        default=None,
        metadata={
            "name": "sentence2 key for preprocessor",
            "api_support": False,
            "exclude_from_hash": True,            
        },
    )
    label_list: List = field(
        default_factory = lambda: ['True','False'],
        metadata={
            "name": "mapping of numeric predictions to labels (for postprocessor)",
            "api_support": False,
            "exclude_from_hash": True,            
        },        
    )

    def __post_init__(self):
        # Placeholder variables
        self._loaded_model = None
        self._tokenizer = None
        self._preprocessor = None
    #    self._scorer_type_as_enum = None
        self._data_collector = None

    def __hash__(self) -> int:
        # Step 1: Identify all fields to be included in the hash
        hashable_fields = [
            k
            for k, v in self.__class__.__dataclass_fields__.items()
            if not "exclude_from_hash" in v.metadata
            or not v.metadata["exclude_from_hash"]
        ]

        # Step 2: Run
        return hash(
            f"{self.__class__.__name__}::{json.dumps({k: v for k, v in vars(self).items() if k in hashable_fields }, sort_keys=True)}"
        )

    def load(self, *args, **kwargs):
        task_heads = EXTRACTIVE_HEAD
        # Load configuration for model
        config = AutoConfig.from_pretrained(self.model)

        # Initialize tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            use_fast=self.use_fast,
            config=config,
        )
    

        config.sep_token_id = self._tokenizer.convert_tokens_to_ids(
            self._tokenizer.sep_token
        )

        self._loaded_model = AutoModelForSequenceClassification.from_pretrained(self.model)        

        self._preprocessor = TextClassifierPreProcessor(
            sentence1_key=self.sentence1_key,
            sentence2_key=self.sentence2_key,
            language_key=None,
            tokenizer=self._tokenizer,
            load_from_cache_file=False,
            max_seq_len=self._tokenizer.model_max_length,
            example_id_key=self.id_key,
            label_list=self.label_list,
            padding=False
        )                   



        # Configure data collector
        self._data_collector = DataCollatorWithPadding(self._tokenizer)

    def _predict(self, input_texts: List[str], context: List[List[str]], *args, **kwargs):
        # Step 1: Locally update object variable values, if provided
        max_num_answers = (
            kwargs["max_num_answers"]
            if "max_num_answers" in kwargs
            else self.max_num_answers
        )

        max_answer_length = (
            kwargs["max_answer_length"]
            if "max_answer_length" in kwargs
            else self.max_answer_length
        )


        postprocessor = TextClassifierPostProcessor(
            k=max_num_answers, 
            drop_label=None,
            n_best_size=self.n_best_size,
            max_answer_length=max_answer_length,
            label_list = self.label_list,
            id_key=self.id_key,
            output_label_prefix=self.output_label_prefix,
        )        


        trainer = NWayTrainer( 
            model=self._loaded_model,
            tokenizer=self._tokenizer,
            data_collator=self._data_collector,
            post_process_function=postprocessor.process,
        )        

        # Step 4: Prepare dataset from input texts and contexts
        eval_examples = Dataset.from_dict(
            dict(
                question=input_texts,
                context=context,
                example_id=[str(idx) for idx in range(len(input_texts))],
            )
        )

        eval_examples, eval_dataset = self._preprocessor.process_eval(eval_examples)

        # Step 5: Run predict
        prediction_output=trainer.predict(eval_dataset, eval_examples)
        return prediction_output


    def apply(self, input_texts: List[str], context: List[List[str]], *args, **kwargs):
        prediction_output = self._predict(input_texts, context, args, kwargs)

        predictions = [[] for _ in range(len(input_texts))]
        for passage_idx, raw_predictions in prediction_output.predictions.items():
            for raw_prediction in raw_predictions:
                processed_prediction = {}
                processed_prediction["example_id"] = raw_prediction["example_id"]
                processed_prediction["span_answer_text"] = raw_prediction[
                    "qtc_pred"
                ]
                processed_prediction["span_answer"] = {'start_position':0, 'end_position':0}
                processed_prediction["confidence_score"] = np.float64(0.0)
                predictions[int(passage_idx)].append(processed_prediction)

        return predictions
