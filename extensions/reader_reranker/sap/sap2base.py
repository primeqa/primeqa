# read in the file and then check if can
# find sentence with high rouge to consider as answer passage in the document.
# Finally, convert to Base format.

import pandas as pd
from primeqa.mrc.metrics.rouge.rouge import ROUGE
import re
import json
import glob

from spacy.lang.en import English

nlp = English()
nlp.add_pipe("sentencizer")

seen_pos = set()
data_pos = []
data_neg = []
questions = []

rouge = ROUGE()

# input_file = "/dccstor/gma/arafat/sap/data/SAP_20230710.ensemble.grounded-and-useful.jsonl"
# input_file = "/dccstor/gma/arafat/sap/data/SAP_20230710.ensemble.not-useful.jsonl"
# input_file = "/dccstor/gma/arafat/sap/data/SAP_20230710.ensemble.not-grounded.jsonl"

files = glob.glob("/dccstor/gma/arafat/sap/data/*grounded-and-useful.jsonl")

threshold = 0.4
low_threshold_count = 0
for input_file in files:
    sap_data = pd.read_json(input_file, lines=True)

    count = 0
    document_length = 0
    num_sentences = 0
    num_paragraphs = 0

    for index, row in sap_data.iterrows():
        output_format = {}
        # 6 or more \n is new section
        # look within for answer - and divide by \n within
        sections = re.split('(\n{5}\n+)', row['document'])
        document_new = ""

        document_offsets_start = []
        document_offsets_end = []
        sentence_offsets_start = []
        sentence_offsets_end = []
        newline_offsets_start = []
        newline_offsets_end = []

        max_rouge = -1.0
        max_sentence = -1.0
        max_rouge_par = -1.0
        max_document = -1.0
        max_document_par = -1.0
        max_rouge_newline = -1.0
        max_newline_index = -1.0
        max_newline_sentence_count = -1
        sentence_index = 0
        document_index = 0
        start = 0
        end = 0

        for section in sections:
            if re.fullmatch('(\n{5}\n+)', section) is not None:
                end = len(section) + start
                document_new += section
                start = len(document_new)
                continue
            section_n = re.split('\n', section)
            sentences = []

            for newline in section_n:
                sents = list(nlp(newline).sents)
                if 'grounded-and-useful' in input_file:
                    hf_metric, _ = rouge._rougel_score(row['response'], newline)
                    if hf_metric > max_rouge_newline:
                        max_rouge_newline = hf_metric
                        max_newline_index = len(sentence_offsets_start) + len(sentences)
                        max_newline_sentence_count = len(sents)
                sentences.extend(sents)
                sentences.append("\n")

            document_offsets_start.append(start)

            if 'grounded-and-useful' in input_file:
                    hf_metric, _ = rouge._rougel_score(row['response'], section)
                    if hf_metric > max_rouge_par:
                        max_rouge_par = hf_metric
                        max_document_par = document_index

            for sentence in sentences:
                sentence_offsets_start.append(start)
                if type(sentence) == str and sentence == "\n":
                    end = len(sentence) + start
                    document_new += sentence
                    sentence_offsets_end.append(end)
                    start = len(document_new)
                    sentence_index += 1
                    continue
                else:
                    end = len(sentence.text) + start
                    document_new += sentence.text
                    sentence_offsets_end.append(end)
                    start = len(document_new)

                if 'grounded-and-useful' in input_file:
                    hf_metric, _ = rouge._rougel_score(row['response'], sentence.text)
                    if hf_metric > max_rouge:
                        max_rouge = hf_metric
                        max_sentence = sentence_index
                        max_document = document_index
                sentence_index += 1
            document_index += 1
            document_offsets_end.append(end)
            
            num_sentences += len(sentences)

        document_length += len(document_new)
        num_paragraphs += len(sentence_offsets_start)

        # base format
        # * 'question': `str`
        # * 'context': `list[str]`
        # * 'example_id': `str`
        # Required for training data:
        # 'target': `{'start_positions': list[int], 'end_positions': list[int], 'passage_indices': list[int], 'yes_no_answer': list[str] }`
        # Required for `single_context_multiple_passages=True`:
        # * 'passage_candidates' : `{ 'start_positions': list[int], 'end_positions': list[int] }`

        output_format['example_id'] = input_file[input_file.rindex("ensemble.")+9:-6] + "_" + str(index)
        output_format['document_id'] = row['document_id']
        output_format['max_rouge'] = max_rouge
        output_format['question'] = row['query']
        output_format['context'] = [document_new]
        output_format['target'] = {} 
        if max_sentence == -1: # no answer
            output_format['target']['start_positions'] = [-1]
            output_format['target']['end_positions'] = [-1]
            output_format['target']['passage_indices'] = [-1]
        elif max_rouge >= max_rouge_par and max_rouge >= max_rouge_newline: # short answer
            if max_rouge < threshold:
                print(f"low threshold: {max_rouge}")
                low_threshold_count += 1
                continue
            output_format['target']['start_positions'] = [sentence_offsets_start[max_sentence]]
            output_format['target']['end_positions'] = [sentence_offsets_end[max_sentence]]
            output_format['target']['passage_indices'] = [max_document]
        elif max_rouge_newline >= max_rouge_par : # short answer multiple sentences
            if max_rouge_newline < threshold:
                print(f"low threshold: {max_rouge_newline}")
                low_threshold_count += 1
                continue
            output_format['target']['start_positions'] = [sentence_offsets_start[max_newline_index]]
            output_format['target']['end_positions'] = [sentence_offsets_end[max_newline_index+max_newline_sentence_count]]
            output_format['target']['passage_indices'] = [max_document]
        else: # long answer
            if max_rouge_par < threshold:
                print(f"low threshold: {max_rouge_par}")         
                low_threshold_count += 1
                continue  
            output_format['target']['start_positions'] = [-1]
            output_format['target']['end_positions'] = [-1]
            output_format['target']['passage_indices'] = [max_document_par]

        output_format['target']['yes_no_answer'] = ["NONE"]
        output_format['passage_candidates'] = {}
        output_format['passage_candidates']['start_positions'] = document_offsets_start
        output_format['passage_candidates']['end_positions'] = document_offsets_end

        if 'grounded-and-useful' in input_file:
            data_pos.append(output_format)
            seen_pos.add(row['document_id'] + row['query'])
        elif row['document_id'] + row['query'] in seen_pos:
            continue
        else:
            data_neg.append(output_format)
            questions.append(row['query'])
        count += 1

        if count % 100 == 0:
            print(count)

    print(input_file)
    print(f"paragraph_length: {document_length/num_paragraphs}")
    print(f"num_sentences: {num_sentences/num_paragraphs}")
    print(f"doc length: {document_length/len(sap_data)}")
    print(f"num paragraphs: {num_paragraphs/len(sap_data)}")
    print(f"skip threshold: {low_threshold_count}")

# shuffle questions with passages and set answer to empty with answer_start=-1
# import random
# random.shuffle(questions)

# for line in data_neg:
#     question = questions.pop()

#     if line['question'] == question:
#         question = questions.pop()
#         questions.append(line['question'])
#     line['question'] = question
#     # line['answers']['text'] = [""]
#     # line['answers']['answer_start'] = [-1]

with open("/dccstor/srosent3/reranking/sap_genq/nq_format_new/train/synthetic_questions_baseformat_pos_train.jsonl", "w", encoding='utf-8') as outfile:
    for line in data_pos[:-1000]:
        outfile.write(json.dumps(line) + "\n")

with open("/dccstor/srosent3/reranking/sap_genq/nq_format_new/dev/synthetic_questions_baseformat_pos_dev.jsonl", "w", encoding='utf-8') as outfile:
    for line in data_pos[-1000:]:
        outfile.write(json.dumps(line) + "\n")

# with open("/dccstor/srosent3/reranking/sap_genq/more_neg/synthetic_questions_squadformat_neg.jsonl", "w", encoding='utf-8') as outfile:
#     for line in data_neg:
#         outfile.write(json.dumps(line) + "\n")