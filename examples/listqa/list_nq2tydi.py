# this code will get the NQ answers that contain lists and convert the lists from NQ format to TyDi format. 
# Note: This is a heuristic and there may be some non-list questions. 
# A question is considered to have a list as an answer if there is a list in the paragraph and no short answer.
# The paragraph offsets are used as the "short answer".

import json
import gzip
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from tabnanny import verbose
from transformers import HfArgumentParser
import sys, os
from tqdm import tqdm
import glob

@dataclass
class List2TyDiArguments:
    """
    Arguments pertaining to processing nq.
    """
    input_file: str = field(
        default='/NQ/download/dir/here', metadata={"help": "input file(s) of nq data in original format."}
    )
    output_file: str= field(default='/output/loc/here/jsonl', metadata={"help": "directory to output file(s) in tydi google format. (jsonl)"}
    )
    num_lines: int = field(default=-1, metadata={"help":"number of examples in the dataset input file (7830 for NQ dev), -1 if unknown or multiple files at once."}
    )
    verbose: bool = field(
        default=False, metadata={"help": "Print full examples"}
    )

class ListNQSubset:
    """
    Class to process the full NQ dataset and get the list subset

    main function is process which requires the following inputs:

    input_file = location of NQ dataset file(s)
    output_file = location for output
    num_lines = number of lines to read (-1 for all)
    verbose = verbose printing for debugging
    """

    def __init__(self) -> None:
        self._LIST_TAGS = {'<ol', '<ul', '<dl'}
        self.keep_list = False
        self.keep_sa = True
        self.avoid_overlap = False
        pass

    @staticmethod
    def load_json_from_file(file_name, num_lines=-1):
        """
        Load dataset from jsonl.gz files 
        """
        data = []

        logging.info(file_name)
        if num_lines == -1:
            num_lines = sum(1 for line in gzip.open(file_name, 'rt', encoding='utf-8'))
        with gzip.open(file_name, 'rt', encoding='utf-8') as f:
            for i in tqdm(range(num_lines)):
                line = f.readline()
                if len(line.strip()) > 0:
                    data.append(json.loads(line))
        return data

    @staticmethod
    def drop_html_tokens(document_tokens):
        """
        Drop html tokens from NQ document
        """        
        no_html_document_tokens = []
        document_plaintext = ""
        doc_offset = 0
        for token in document_tokens:
            if token['html_token'] and token['token'].lower() == "<li>":
                token['token'] = "*"
                # token['end_byte'] = token['start_byte'] + 1
            elif token['html_token']:
                no_html_document_tokens.append(None)
                continue
            token['new_start_byte'] = doc_offset
            token['new_end_byte'] = doc_offset + len(token['token'].encode())
            doc_offset += len(token['token'].encode()) + 1
            document_plaintext += token['token'] + " "
            no_html_document_tokens.append(token)
        return no_html_document_tokens, document_plaintext

    @staticmethod
    def drop_html_tokens_from_span(original_span, document_tokens):
        """
        Drop html token from NQ answer spans 
        """
        # no answer
        if original_span['start_token'] == -1 and original_span['end_token'] == -1:
            return -1, -1
        first_non_html_token = original_span['start_token']
        last_non_html_token = min(len(document_tokens), original_span['end_token'] - 1)
        while first_non_html_token <= last_non_html_token:
            if document_tokens[first_non_html_token] != None and document_tokens[last_non_html_token] != None:
                break
            else:
                if document_tokens[first_non_html_token] == None:
                    first_non_html_token += 1
                if document_tokens[last_non_html_token] == None:
                    last_non_html_token -= 1
        else:
            logging.info("We're going to lose this non null short answer span %s because we couldn't"
                            " find start token index that isn't a html token" % original_span)
            return "Null"

        return document_tokens[first_non_html_token]['new_start_byte'], document_tokens[last_non_html_token]['new_end_byte']


    def get_annotations(self, annotations, html_tokens, text_tokens, updated_indices):
        """
        Get NQ annotations and return in TyDi format if its a list answer. 
        An answer is a list answer if the long answer starts with a list tag and the short answer is null
        """
        is_list_count = 0
        is_sa_count = 0

        tydi_annotations = []
        for annotation in annotations:

            # confirm this answer is a list
            long_span_start_token = annotation['long_answer']['start_token']
            is_list = False
            if html_tokens[long_span_start_token]['html_token']:
                if html_tokens[long_span_start_token]['token'] == "*":
                    text = ""
                    answer_tokens = html_tokens[long_span_start_token:annotation['long_answer']['end_token']]
                    for t in answer_tokens:
                        text += t['token'] + " "
                    if len(text.split("*")) > 2:
                        is_list = True
                else:
                    for list_tag in self._LIST_TAGS:
                        if html_tokens[long_span_start_token]['token'].lower().startswith(list_tag):
                            is_list = True
                            break

            if is_list and len(annotation['short_answers']) == 0:
                is_list_count += 1
                if self.keep_list:
                    start_byte, end_byte = self.drop_html_tokens_from_span(annotation['long_answer'],text_tokens)
                else:
                    continue
            elif self.keep_sa and len(annotation['short_answers']) == 0:
                start_byte = end_byte = -1
            elif self.keep_sa:
                is_sa_count += 1
                start_byte, end_byte = self.drop_html_tokens_from_span(annotation['short_answers'][0],text_tokens)
            else:
                continue

            if end_byte < start_byte:
                logging.Error("error! end < start " + str(start_byte) + "," + str(end_byte))
            ## non-lists end
                    
            minimal_answer = {"plaintext_start_byte":start_byte,"plaintext_end_byte":end_byte}
            passage_answer = {"candidate_index":updated_indices[annotation['long_answer']['candidate_index']]}

            tydi_annotations.append({'annotation_id':str(annotation['annotation_id']), 'minimal_answer': minimal_answer, 'passage_answer': passage_answer, 'yes_no_answer':annotation['yes_no_answer']})
        
        # may want to change this, but for now if its in the list subset, don't put in the sa subset (even if both types of answers)
        if self.avoid_overlap and is_list_count >= 1 and self.keep_sa and not self.keep_list:
            logging.info("list and short answer")
            return []
        return tydi_annotations

    def get_paragraphs(self, candidates_json, text_tokens):
        """
        Get NQ paragraphs and return in TyDi format 
        """
        updated_indices = [0] * len(candidates_json)
        passage_answer_candidates=[]

        count = 0
        i = 0
        for candidate_json in candidates_json:
            i += 1
            start_token = candidate_json['start_token']
            while text_tokens[start_token] == None and start_token < candidate_json['end_token']:
                start_token+=1
            end_token = candidate_json['end_token']
            while text_tokens[end_token] == None and end_token > candidate_json['start_token']:
                end_token-=1

            
            if end_token <= start_token:
                continue
            else:
                updated_indices[i-1] = count
                count += 1
                passage_answer_candidates.append({'plaintext_start_byte':text_tokens[start_token]['new_start_byte'],
            'plaintext_end_byte':text_tokens[end_token]['new_end_byte']})
                                                
        return passage_answer_candidates, updated_indices


    def nq2tydi(self, example, verbose=False):
        """
        Convert NQ to TyDi format 
        """

        tydi_format = {}                            
        # cleanup html
        text_tokens, document_plaintext = self.drop_html_tokens(example['document_tokens'])
        tydi_format['document_plaintext'] = document_plaintext
        tydi_format['document_title'] = example['document_title']
        tydi_format['document_url'] = example['document_url']
        tydi_format['example_id'] = str(example['example_id'])
        tydi_format['language'] = 'english'
        tydi_format['question_text'] = example['question_text']
        tydi_format['passage_answer_candidates'], updated_indices = self.get_paragraphs(example['long_answer_candidates'], text_tokens)
        tydi_format['annotations'] = self.get_annotations(example['annotations'], example['document_tokens'], text_tokens, updated_indices)
        if len(tydi_format['annotations']) == 0:
            return None
        if verbose:
            if tydi_format['annotations'][0]['minimal_answer']['plaintext_start_byte'] != -1:
                logging.info(example['example_id'])
                logging.info(example['question_text'])
                logging.info(document_plaintext.encode('utf-8')[tydi_format['annotations'][0]['minimal_answer']['plaintext_start_byte']:tydi_format['annotations'][0]['minimal_answer']['plaintext_end_byte']])
                logging.info("-----------------")
        return tydi_format

    def process(self, input_file, output_file, num_lines=-1, verbose=False):
        """
        Process NQ dataset to get list answers and return in TyDi format 
        """
        avg_gold_length = 0
        avg_list_length = 0
        count = 0

        logger = logging.getLogger(__name__)
        logging.getLogger().setLevel(logging.INFO)

        logging.info("Loading NQ data...")
        files = glob.glob(input_file)

        for file in files:
            nq_data = self.load_json_from_file(file, num_lines)
            logging.info("Done")
            tydi_data = []

            logging.info("Converting to tydi")
            for example in tqdm(nq_data):
                tydi_example = self.nq2tydi(example, verbose)
                if tydi_example == None:
                    continue
                for annotation in tydi_example['annotations']:
                    avg_gold_length += annotation['minimal_answer']['plaintext_end_byte'] - annotation['minimal_answer']['plaintext_start_byte']
                    avg_list_length += tydi_example['document_plaintext'][annotation['minimal_answer']['plaintext_start_byte']:annotation['minimal_answer']['plaintext_end_byte']].count("*")
                    count += 1
                tydi_data.append(tydi_example)
            logging.info("data size: " + str(len(tydi_data)))
            with open(output_file,'ab') as writer:
                for data in tydi_data:
                    writer.write((json.dumps(data) + "\n").encode())

        logging.info("count: " + str(count))
        logging.info("average answer length: " + str(avg_gold_length/count))
        logging.info("average list length: " + str(avg_list_length/count))

def main():

    parser = HfArgumentParser(List2TyDiArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()[0]
    list_processor = ListNQSubset()
    list_processor.process(args.input_file, args.output_file, args.num_lines, args.verbose)

if __name__ == "__main__":
    main()