# In this file download TyDi HF dataset and save the question and gold answer passage in csv format
# for all boolean questions for training and for all questions from dev:
# example_id, question, language, answer

import csv
from copy import deepcopy
from dataclasses import dataclass, field
from distutils.command.config import config
import logging
from transformers import HfArgumentParser
import sys, os
from datasets import load_dataset
from typing import Optional, Type

@dataclass
class BoolTyDiCSVArguments:
    """
    Arguments pertaining to processing nq.
    """
    output_dir: str= field(default='./examples/boolqa/', metadata={"help": "directory to output file(s) in csv format"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    lower_case: bool = field(
        default=False, 
        metadata={"help": "lowercase all text"}
    )
class BoolTyDiSubset:
    """
    Class to process the full NQ dataset and get the list subset

    main function is process which requires the following inputs:

    input_file = location of NQ dataset file(s)
    output_file = location for output
    num_lines = number of lines to read (-1 for all)
    verbose = verbose printing for debugging
    """

    def __init__(self) -> None:
        pass

    def get_data(self, example, lower_case=False):
        question_text = example['question_text']
        passage_candidate_index = example['annotations']['passage_answer_candidate_index'][0]
        start_byte = example['passage_answer_candidates']['plaintext_start_byte'][passage_candidate_index]
        end_byte = example['passage_answer_candidates']['plaintext_end_byte'][passage_candidate_index]
        passage_text = example['document_plaintext'].encode('utf-8')[start_byte:end_byte]
        label = example['annotations']['yes_no_answer'][0]
        if lower_case:
            question_text = question_text.lower()
            passage_text = passage_text.lower()
        return question_text, passage_text.decode('utf-8').replace("\n"," "), label

    def get_writers(self, output_dir, source):
        qtype_file = open(output_dir + "/qtype_" + source + ".csv", 'w')
        evc_file = open(output_dir + "/evidence_span_" + source + ".csv", 'w')
        qtype_writer = csv.writer(qtype_file, quoting=csv.QUOTE_MINIMAL)
        evidence_span_writer = csv.writer(evc_file, quoting=csv.QUOTE_MINIMAL)
        qtype_writer.writerow(["example_id", "question","language","label"])
        evidence_span_writer.writerow(["example_id", "question", "language", "label", "passage"])
        return qtype_file, evc_file, qtype_writer, evidence_span_writer

    def process(self, output_dir, cache_dir, lower_case):
        dataset = load_dataset("tydiqa", "primary_task", cache_dir=cache_dir)
        
        count = 0
        # train
        qtype_file, evc_file, qtype_writer, evidence_span_writer = self.get_writers(output_dir, "train")
        for example in dataset['train']:
            count += 1
            # skip NA since we don't know if they are boolean or short answer
            if example['annotations']['passage_answer_candidate_index'][0] == -1 or \
             (example['annotations']['minimal_answers_start_byte'][0] == -1 and example['annotations']['yes_no_answer'][0] == 'NONE'):
                continue
            question_text, passage_text, label = self.get_data(example, lower_case=lower_case)
            if label == "YES" or label == "NO":
                qtype_writer.writerow([str(count), question_text,example['language'],"boolean"])
            else:
                qtype_writer.writerow([str(count), question_text,example['language'],"other"])
            evidence_span_writer.writerow([str(count), question_text, example['language'], label, passage_text])
        qtype_file.close()
        evc_file.close()
        # dev and eval
        qtype_file, evc_file, qtype_writer, evidence_span_writer = self.get_writers(output_dir, "dev")
        qtype_file_e, evc_file_e, qtype_writer_e, evidence_span_writer_e = self.get_writers(output_dir, "eval")
        for example in dataset['validation']:
            count += 1
            question_text, passage_text, label = self.get_data(example,lower_case=lower_case)
            is_dev = True
            # skip NA since we don't know if they are boolean or short answer
            if example['annotations']['passage_answer_candidate_index'][0] == -1 or \
             (example['annotations']['minimal_answers_start_byte'][0] == -1 and example['annotations']['yes_no_answer'][0] == 'NONE'):
                is_dev = False
            if label == "YES" or label == "NO":
                if is_dev:
                    qtype_writer.writerow([str(count), question_text,example['language'],"boolean"])
                qtype_writer_e.writerow([str(count), question_text,example['language'],"boolean"])
            else:
                if is_dev:
                    qtype_writer.writerow([str(count), question_text,example['language'],"other"])
                qtype_writer_e.writerow([str(count), question_text,example['language'],"other"])
            if is_dev:
                evidence_span_writer.writerow([str(count), question_text, example['language'], label, passage_text])
            evidence_span_writer_e.writerow([str(count), question_text, example['language'], label, passage_text])
        qtype_file.close()
        evc_file.close()
        qtype_file_e.close()
        evc_file_e.close()

def main():

    parser = HfArgumentParser(BoolTyDiCSVArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()[0]
    bool_processor = BoolTyDiSubset()
    bool_processor.process(args.output_dir, args.cache_dir, args.lower_case)

if __name__ == "__main__":
    main()