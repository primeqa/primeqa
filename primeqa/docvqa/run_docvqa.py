import os
import json
import pandas as pd

import logging
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from primeqa.docvqa.models.docvqa_model import DocVQAModel
from primeqa.docvqa.metrics.answer_accuracy import compute_anls_score


@dataclass
class DocVQAArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune/evaluate from.
    """
    dev_data_path: str = field(
       metadata={"help": "Dev data path for evalution on user's own dataset"}
    )

    dataset_name: str = field(
       default='docvqa', metadata={"help": "Name of the dataset to evaluate the  model on"}
    )

# modified from
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to for evaluation.
    """

    model_name_or_path: str = field(
        default="impira/layoutlm-document-qa", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class TrainingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to for evaluation.
    """
        
    output_dir: str = field(
        default="/tmp/", metadata={"help": "Path to the output directory"}
    )
    
    do_train: bool = field(
        default=False,
        metadata={"help": "Trains a custom model if true"}
    ) 

    do_eval: bool = field(
        default=False,
        metadata={"help": "Evaluate a fine-tuned model if true"}
    ) 

def run_docvqa(data_args, model_args, training_args):
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    model = DocVQAModel(model_args.model_name_or_path)

    if data_args.dataset_name != "docvqa":
        raise NotImplementedError("Evaluation is supported only for the Docvqa dataset.")

    if training_args.do_train:
        raise NotImplementedError("Training is not supported for Docvqa.")

    if training_args.do_eval:
        for f in os.listdir(data_args.dev_data_path):
            if f.endswith("json"):
                annotations_file = os.path.join(data_args.dev_data_path, f)
            else:
                ocr_output = os.path.join(data_args.dev_data_path, f)
            
        with open(annotations_file) as jfp:
            data = json.load(jfp)
    
        annotations = data['data'] # type = list
        gold_answers = []
        predicted_answers = []
        logger.info("*** Getting Predictions ***")
        for annotation in tqdm(annotations, desc="Evaluating the model"):
            # type = dict
            questionId = annotation['questionId'] # type = int
            question = annotation['question'] # type = str
            image = annotation['image'] # type = str
            basename = os.path.basename(image)
            docId = annotation['docId'] # type = int
            answers = annotation['answers'] # type = list
            
            tesseract_json = os.path.join("%s/%s.json" % (ocr_output, basename[:-4]))
            sample = (tesseract_json, [question])
            output = model.predict([sample])
            _, prediction = output[0].popitem()
            predicted_answers.append(prediction)
            gold_answers.append(answers)
        logger.info("*** Scoring Predictions ***")
        anls_score = compute_anls_score(predicted_answers, gold_answers)
        logger.info("ANLS Score: %s" % (anls_score.get("ANLS Score", 0.)))

def main():
    parser = HfArgumentParser((ModelArguments, DocVQAArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    run_docvqa(data_args, model_args, training_args)

if __name__ == "__main__":
    main()
