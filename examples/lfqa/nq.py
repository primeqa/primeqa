# this code will get the NQ answers that contain lists and convert the lists from NQ format to TyDi format. 
# Note: This is a heuristic and there may be some non-list questions. 
# A question is considered to have a list as an answer if there is a list in the paragraph and no short answer.
# The paragraph offsets are used as the "short answer".

import json
import gzip
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from transformers import HfArgumentParser
import sys, os
from tqdm import tqdm
import glob

@dataclass
class LFQAArguments:
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

class LFQANQSubset:
    """
    Class to process the full NQ dataset and get the list subset

    main function is process which requires the following inputs:

    input_file = location of NQ dataset file(s)
    output_file = location for output
    num_lines = number of lines to read (-1 for all)
    verbose = verbose printing for debugging
    """

    def __init__(self) -> None:
        self._LIST_TAGS = {'<ol', '<ul', '<dl', '<li', '<dd', '<dt'}
        self._TABLE_TAGS = {'<td', '<table', '<th', '<tr'}
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
            # if token['html_token'] and token['token'].lower() == "<li>":
            #     token['token'] = "*"
            #     # token['end_byte'] = token['start_byte'] + 1
            # el
            if token['html_token']:
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


    def get_annotations(self, annotations, html_tokens, document_plaintext, question_text, updated_indices):
        """
        To Start check if answer is a long answer with no short answer.
        Skip null long answers
        """
        tydi_annotations = []
        a_type = []
        for annotation in annotations:
            
            # skip short answers
            if len(annotation['short_answers']) != 0 and len(html_tokens[annotation['short_answers'][0]['start_token']:annotation['short_answers'][-1]['end_token']]) < 10:
                short_answer_text = html_tokens[annotation['short_answers'][0]['start_token']:annotation['short_answers'][-1]['end_token']]
                if len(annotation['short_answers']) > 4:
                    logging.info("Short Answer that is slightly long")
                    logging.info(question_text)
                    tokens = ""
                    for token in short_answer_text:
                        tokens += token['token'] + " "
                    logging.info(tokens)
                    logging.info(len(tokens))
                a_type.append("sa")
                continue
            # skip null long answers
            if annotation['long_answer']['candidate_index'] == -1:
                if len(annotation['short_answers']) != 0:
                    logging.info("Short Answer w/o LA")
                    logging.info(question_text)
                    tokens = ""
                    for token in short_answer_text:
                        tokens += token['token'] + " "
                    logging.info(tokens)
                    logging.info(len(tokens))
                a_type.append("null")
                continue

            # check type of long answer
            long_span_start_token = annotation['long_answer']['start_token']
            is_list = False
            is_table = False
            is_boolean = False

            if annotation['yes_no_answer'] != "NONE":
                is_boolean = True
            elif html_tokens[long_span_start_token]['html_token']:
                if not html_tokens[long_span_start_token]['token'].lower().startswith("<p"):
                    for list_tag in self._LIST_TAGS:
                        if html_tokens[long_span_start_token]['token'].lower().startswith(list_tag):
                            is_list = True
                            break
                    if not is_list:
                        for table_tag in self._TABLE_TAGS:
                            if html_tokens[long_span_start_token]['token'].lower().startswith(table_tag):
                                is_table = True
                                break
                    if not is_table and not is_list:
                        logging.info("other html token: " + html_tokens[long_span_start_token]['token'].lower())
                        
            if is_boolean:
                a_type.append("boolean")
            if is_list:
                a_type.append("list")
            elif is_table:
                a_type.append("table")
            else:
                a_type.append("la")
           
            if len(annotation['short_answers']) != 0:
                if is_boolean:
                    logging.info("Boolean with short answer (evidence??)")

                end_token = annotation['short_answers'][-1]['end_token']
                start_token = annotation['short_answers'][0]['start_token']
                while 'new_end_byte' not in html_tokens[end_token]:
                     end_token = end_token - 1
                # if end_token - annotation['short_answers'][0]['start_token'] < 10:
                #     minimal_answer = {"plaintext_start_byte":-1,"plaintext_end_byte":-1}
                # else:
                minimal_answer = {"plaintext_start_byte":html_tokens[start_token]['new_start_byte'],"plaintext_end_byte":html_tokens[end_token]['new_end_byte']}
            else:
                minimal_answer = {"plaintext_start_byte":-1,"plaintext_end_byte":-1}
            passage_answer = {"candidate_index":updated_indices[annotation['long_answer']['candidate_index']]}

            tydi_annotations.append({'annotation_id':annotation['annotation_id'], 'minimal_answer': minimal_answer, 'passage_answer': passage_answer, 'yes_no_answer':annotation['yes_no_answer']})
        return tydi_annotations, a_type

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


    def lfqa(self, example, verbose=False):
        """
        Convert NQ to TyDi format 
        """
        tydi_format = {}                            
        # cleanup html
        text_tokens, document_plaintext = self.drop_html_tokens(example['document_tokens'])
        tydi_format['document_plaintext'] = document_plaintext
        tydi_format['document_title'] = example['document_title']
        tydi_format['document_url'] = example['document_url']
        tydi_format['example_id'] = example['example_id']
        tydi_format['language'] = 'english'
        tydi_format['question_text'] = example['question_text']
        tydi_format['passage_answer_candidates'], updated_indices = self.get_paragraphs(example['long_answer_candidates'], text_tokens)
        tydi_format['annotations'], a_type = self.get_annotations(example['annotations'], example['document_tokens'], document_plaintext, tydi_format['question_text'], updated_indices)
        tydi_format['type'] = a_type
        if len(tydi_format['annotations']) == 0:
            return None, a_type
        if verbose:
            if "sa" not in a_type:
                logging.info(a_type)
                logging.info(example['example_id'])
                logging.info(example['question_text'])
                logging.info("-----------------")
        return tydi_format, a_type
        

    def process(self, input_file, output_file, num_lines=-1, verbose=False):
        """
        Process NQ dataset to get list answers and return in TyDi format 
        """
        list_count = 0
        count = 0

        logger = logging.getLogger(__name__)
        logging.getLogger().setLevel(logging.INFO)

        logging.info("Loading NQ data...")
        files = glob.glob(input_file)

        answer_types = {"list" : 0, "table" : 0, "la": 0, "null": 0, "sa": 0, "boolean": 0}

        for file in files:
            nq_data = self.load_json_from_file(file, num_lines)
            logging.info("Done")
            tydi_data = []

            logging.info("Converting to tydi")
            for example in tqdm(nq_data):
                item, a_types = self.lfqa(example, verbose)
                if item is not None:
                    tydi_data.append(item)
                has_list = False
                for a_type in a_types:
                    if a_type == "list":
                        has_list = True
                    answer_types[a_type] += 1
                if has_list:
                    list_count += 1
            logging.info("data size: " + str(len(nq_data)))
            count += len(tydi_data)
            logging.info("lfqa answer types: " + str(answer_types))
            logging.info("lfqa count: " + str(count))
            logging.info("list count: " + str(list_count))
            with open(output_file,'ab') as writer:
                for data in tydi_data:
                    writer.write((json.dumps(data) + "\n").encode())

        logging.info("lfqa answer types: " + str(answer_types))
        logging.info("lfqa count: " + str(count))
        logging.info("list count: " + str(list_count))


def main():

    parser = HfArgumentParser(LFQAArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()[0]
    list_processor = LFQANQSubset()
    list_processor.process(args.input_file, args.output_file, args.num_lines, args.verbose)

if __name__ == "__main__":
    main()