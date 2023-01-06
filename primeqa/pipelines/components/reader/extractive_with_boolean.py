from typing import List
from dataclasses import dataclass, field
import json
import numpy as np

from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset

from primeqa.pipelines.components.base import ReaderComponent
from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from primeqa.mrc.models.task_model import ModelForDownstreamTasks
from primeqa.mrc.processors.preprocessors.base import BasePreProcessor
from primeqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from primeqa.mrc.trainers.mrc import MRCTrainer
from primeqa.pipelines.components.reader.text_classifier_reader import TextClassifierReader
from primeqa.pipelines.components.reader.extractive import ExtractiveReader
from primeqa.boolqa.score_normalizer.score_normalizer import ScoreNormalizer


@dataclass
class ExtractiveWithBooleanReader(ReaderComponent):
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

    boolean_config: str = field(
        default="/store/models/tydi_boolqa_config.json",
        metadata={"name": "Model", "api_support": True},
    )
    model: str = field(
#        default="/dccstor/jsmc-nmt-01/bool/expts/leaderboard/mrc/a4_1e-5_1_42_a100/",
        default="/store/models/a4_1e-5_1_42_a100/",
        metadata={"name": "Model", "api_support": True},
    )    
    use_fast: bool = field(
        default=True,
        metadata={
            "name": "Use the fast version of the tokenizer",
            "options": [True, False],
        },
    )
    stride: int = field(
        default=128,
        metadata={
            "name": "Stride",
            "description": "Step size to move sliding window across context",
            "range": [8, 256, 8],
        },
    )
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
    scorer_type: str = field(
        default=SupportedSpanScorers.WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF.value,
        metadata={
            "name": "Scoring algorithm",
            "options": [
                SupportedSpanScorers.SCORE_DIFF_BASED.value,
                SupportedSpanScorers.TARGET_TYPE_WEIGHTED_SCORE_DIFF.value,
                SupportedSpanScorers.WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF.value,
            ],
        },
    )
    min_score_threshold: float = field(
        default=None,
        metadata={
            "name": "Minimum score threshold",
            "api_support": True,
            "exclude_from_hash": True,
        },
    )

    def __post_init__(self):
        print('in __post_init__')
        # Placeholder variables
        self._loaded_model = None
        self._tokenizer = None
        self._preprocessor = None
        self._scorer_type_as_enum = None
        self._data_collector = None
        self._scoreNormalizer = None

        self._extractiveReader = ExtractiveReader()
        self._booleanQTCReader = TextClassifierReader()
        self._booleanEVCReader = TextClassifierReader()
        self._extractiveReader.__post_init__()
        self._booleanQTCReader.__post_init__()
        self._booleanEVCReader.__post_init__()



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
        # TODO this is restricted to file system
        boolean_config=json.load(open(self.boolean_config))
        qtc_config=boolean_config['qtc']
        evc_config=boolean_config['evc']
        sn_config=boolean_config['sn']
        # TODO this in config file?
        # dispatch parameters to the underlying extractiveReader        
        mrc_config_dict={ k:getattr(self,k) for k in self.__class__.__dataclass_fields__.keys() }

        self._extractiveReader.init_from_dict(mrc_config_dict)
        self._extractiveReader.load(args, kwargs)
        self._booleanQTCReader.load(args, **qtc_config)
        self._booleanEVCReader.load(args, **evc_config)

        self._scoreNormalizer = ScoreNormalizer(sn_config['model_name_or_path'])
        self._scoreNormalizer.load_model()        

    def apply(self, input_texts: List[str], context: List[List[str]], *args, **kwargs):
        min_score_threshold = (
            kwargs["min_score_threshold"]
            if "min_score_threshold" in kwargs
            else self.min_score_threshold
        )


        predict_output=self._extractiveReader._predict(input_texts, context, args, kwargs)

        qtc_prediction_output=self._booleanQTCReader._predict(input_texts, context, args, kwargs)
        evc_prediction_output=self._booleanEVCReader._predict(input_texts, context, args, kwargs)

        qtc_pred_key=self._booleanQTCReader.output_label_prefix+"_pred"
        evc_pred_key=self._booleanEVCReader.output_label_prefix+"_pred"

        predictions = [[] for _ in range(len(input_texts))]

        for passage_idx, raw_predictions in predict_output.items():
            print(json.dumps(raw_predictions, indent=4))
            qtcp = qtc_prediction_output.predictions[raw_predictions[0]['example_id']] [0][qtc_pred_key]
            evcp = evc_prediction_output.predictions[raw_predictions[0]['example_id']] [0][evc_pred_key]            
            for idx, raw_prediction in enumerate(raw_predictions):
                processed_prediction = {}
                processed_prediction["example_id"] = raw_prediction["example_id"]
                mrcp = raw_prediction['span_answer_text']
                print(idx)
                if idx==0:
                    processed_prediction["span_answer_text"] = f'question type: {qtcp} boolean answer: {evcp} mrc: {mrcp}'
                else:
                    processed_prediction["span_answer_text"] = mrcp
                processed_prediction["span_answer"] = raw_prediction["span_answer"]
                # processed_prediction["confidence_score"] = raw_prediction[
                #    "confidence_score"
                # ]
                # handle score normalizer here
                processed_prediction["confidence_score"] = self._handle_boolean_score_normalizer( raw_prediction, qtcp=="boolean")


                predictions[int(passage_idx)].append(processed_prediction)


        # Step 6: If min_score_threshold is provide, use it to filter out predictions
        if min_score_threshold:
            filtered_predictions = []
            for sorted_predictions_for_passage in predictions:
                filtered_predictions_for_passage = []
                for sorted_prediction in sorted_predictions_for_passage:
                    if sorted_prediction["confidence_score"] >= min_score_threshold:
                        filtered_predictions_for_passage.append(sorted_prediction)

                filtered_predictions.append(filtered_predictions_for_passage)
            return filtered_predictions
        else:
            return predictions


        return predictions

    def _handle_boolean_score_normalizer(self, qa_pred: dict, question_label : int ) -> float :
        b_score = qa_pred['start_logit']
        e_score = qa_pred['end_logit']
        na_score = qa_pred['target_type_logits'][0] if 'target_type_logits' in  qa_pred else 0.0
        #question_label = 1 if qa_pred['question_type_pred'] == qtc_is_boolean_label else 0
        feature_list = [question_label,b_score,e_score,na_score]
        features = np.array(feature_list).reshape(1, -1)
        new_score = self._scoreNormalizer._model.predict_proba(features)[0][1]
        return new_score



