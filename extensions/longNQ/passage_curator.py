# open the tydi DEV and TEST files of LongNQ and save all of the passages. Keep unique IDs per passage and do not store duplicates.
# store as <id>\t<text>\t<title>\n

import pandas as pd
import glob
from rouge_score import rouge_scorer

rouge = rouge_scorer.RougeScorer(rouge_types=['rouge1',], split_summaries=False)

do_corpus = False
do_questions = True
unique_passages = None

if do_corpus:
    # ensure priority of ids - test -> dev -> train
    data_files = glob.glob("/dccstor/srosent2/generative/appen/final/original_tydi/test/*.jsonl")
    data_files.extend(glob.glob("/dccstor/srosent2/generative/appen/final/original_tydi/dev/*.jsonl"))
    data_files.extend(glob.glob("/dccstor/srosent2/generative/appen/final/original_tydi/train/*.jsonl"))

    dfs = []

    for file_name in data_files:
        data = pd.read_json(file_name, lines=True, orient='records', dtype={'example_id':str})
        if 'train' in file_name:
            data['split'] = 'train'
        elif 'dev' in file_name:
            data['split'] = 'dev'
        else:
            data['split'] = 'test'
        dfs.append(data)

    all_data = pd.concat(dfs, ignore_index=True)

    data_by_title = {}

    count = 0
    more_passages = 0
    duplicates = 0
    passage_lengths=[20000,0,0,0]
    num_passages = 0
    multiple_gold_passages = 0

    def compute_length(passage_text):
        passage_len = len(passage_text.decode().split(" "))
        passage_lengths[1] += passage_len
        if passage_len < passage_lengths[0]:
            passage_lengths[0] = passage_len
        if passage_len > passage_lengths[2]:
            passage_lengths[2] = passage_len
        passage_lengths[3]+=1

    for i, row in all_data.iterrows():
        count += len(row['passage_answer_candidates'])

        answer_loc = set()
        for ann in row['annotations']:
            if ann['passage_answer']['candidate_index'] != -1:
                answer_loc.add(ann['passage_answer']['candidate_index'])     
        if len(answer_loc) > 1:
            multiple_gold_passages += 1
        num_passages += len(answer_loc)  
                
        if row['document_title'] in data_by_title:
            duplicates += 1

            if row['document_url'][row['document_url'].rindex("=")+1:] !=  data_by_title[row['document_title']]['id']:
                print(f"New ID for {row['document_title']}")
            data_by_title[row['document_title']]['id'].add(row['document_url'][row['document_url'].rindex("=")+1:])
            
            index = -1
            for candidate in row['passage_answer_candidates']:
                index += 1
                if index not in answer_loc:
                    continue

                added = False
                passage_text = bytes(row['document_plaintext'],'utf-8')[candidate['plaintext_start_byte']:candidate['plaintext_end_byte']]
                compute_length(passage_text)
                
                # if passage_text not in data_by_title[row['document_title']]['passages'].values():
                # duplicate offsets are super similar so just keep the longer one instead of two versions.
                more_passages += 1
                if f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}" in data_by_title[row['document_title']]['passages']:
                    if data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"]['passage_text'] != passage_text:
                        rouge_score = rouge.score(data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"]['passage_text'],passage_text)['rouge1'][2]
                        print(f"duplicate offsets - similar gold passage ({rouge_score}). keep this one and not the other")
                        if rouge_score < .98:
                            print(data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"]['passage_text'])
                            print(passage_text)
                            print("----")
                        data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"]['passage_text'] = passage_text
                    else:
                        print("duplicate offsets - same gold passage")
                    data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"]['example_id'].append(row['example_id'])
                    data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"]['split'].add(row['split'])
                else:
                    # these might be similar to other passages - need to check.
                    matching_passage_id = None
                    for passage_id in data_by_title[row['document_title']]['passages']:
                        rouge_score = rouge.score(data_by_title[row['document_title']]['passages'][passage_id]['passage_text'],passage_text)['rouge1'][2]
                        if rouge_score > .90:
                            matching_passage_id = passage_id
                            break
                    # delete the close match if no answers associated with it.
                    if matching_passage_id != None:
                        if data_by_title[row['document_title']]['passages'][matching_passage_id]['example_id'] == []:
                            print(f"adding a new gold passage from a different question -- removing the exact/similar ({rouge_score}) passage that is not gold")
                            del data_by_title[row['document_title']]['passages'][matching_passage_id]
                            data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"] = {"passage_text":passage_text,"example_id":[row['example_id']],"split":set([row['split']])}
                        else:
                            print(f"exact/similar ({rouge_score}) passage is gold, keep the longer one.")
                            if rouge_score < 1 and len(passage_text) > len(data_by_title[row['document_title']]['passages'][matching_passage_id]['passage_text']):
                                data_by_title[row['document_title']]['passages'][matching_passage_id]['passage_text'] = passage_text
                            data_by_title[row['document_title']]['passages'][matching_passage_id]['example_id'].append(row['example_id'])
                            data_by_title[row['document_title']]['passages'][matching_passage_id]['split'].add(row['split'])
                    else:
                        print("adding a new gold passage from a different question")
                        data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"] = {"passage_text":passage_text,"example_id":[row['example_id']], "split":set([row['split']])}
            continue
        id = row['document_url'][row['document_url'].rindex("=")+1:]
        title = row['document_title']
        
        passages = {}
        index = 0
        start = -1
        end = -1
        for candidate in row['passage_answer_candidates']:

            if index not in answer_loc and start != -1 and \
                candidate['plaintext_start_byte'] >= start and \
                    candidate['plaintext_end_byte'] <= end:
                index += 1
                # overlapping candidate
                continue
            elif index in answer_loc and start != -1 and \
                candidate['plaintext_start_byte'] >= start and \
                    candidate['plaintext_end_byte'] <= end:
                # remove overlap due to candidate, but keep as start/end
                del passages[f"{start}-{end}"]
                index += 1
                continue

            passage_text = bytes(row['document_plaintext'],'utf-8')[candidate['plaintext_start_byte']:candidate['plaintext_end_byte']]
            start = candidate['plaintext_start_byte']
            end = candidate['plaintext_end_byte']

            compute_length(passage_text)
            if index in answer_loc:
                passages[f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"] = {"passage_text":passage_text,"example_id":[row['example_id']],"split":set([row['split']])}
            else:
                passages[f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"] = {"passage_text":passage_text,"example_id":[],"split":set([row['split']])}
            index += 1

        data_by_title[title] = {'id':set([id]), 'title':title, 'passages': passages} #, 'answers': answers}

    print(f"multiple gold passages: {multiple_gold_passages}")
    print(f"{duplicates} duplicates. more passages needed to be added: {more_passages}")
    passage_lengths[1] = passage_lengths[1]/passage_lengths[3]
    print(f"min, average, max lengths {passage_lengths}")

    unique_passages = {}
    total_passages = 0
    passages_with_questions = 0
    num_questions = 0

    for item in data_by_title:
        for passage in data_by_title[item]['passages']:
            passage_text = data_by_title[item]['passages'][passage]['passage_text']
            num_words = len(passage_text.decode().split(" "))
            # discard really short or really long passages
            if data_by_title[item]['passages'][passage]['example_id'] == [] and \
                num_words < 15 or num_words > 3000:
                continue
            if data_by_title[item]['passages'][passage]['example_id'] != []:
                passages_with_questions += 1
                num_questions += len(data_by_title[item]['passages'][passage]['example_id'])
            if list(data_by_title[item]['id'])[0] in unique_passages:
                print('duplicate id')
            unique_passages[f"{list(data_by_title[item]['id'])[0]}_{passage}"] = {'text':data_by_title[item]['passages'][passage]['passage_text'].decode(), 'title': data_by_title[item]['title'],'example_ids':data_by_title[item]['passages'][passage]['example_id'],'splits':data_by_title[item]['passages'][passage]['split']}

    print(f"{passages_with_questions} passages with {num_questions} questions and {num_passages} selected passages.")
    print(f"Num unique passages: {len(unique_passages)}/{count}")
    # dump passages to tsv
    pd.DataFrame.from_dict(unique_passages, orient='index', columns=["id","text","title", "example_ids", "splits"]).drop(columns=["example_ids","splits"]).to_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/passages.tsv", sep="\t")
    pd.DataFrame.from_dict(unique_passages, orient='index', columns=["id","text","title","example_ids","splits"]).to_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/LongNQ_train_dev_test_passages_wids.tsv", sep="\t")

if do_questions:
    # questions.tsv: <id> <question> <doc-id-list> <answers>
    # load longNQ data - make sep files for train, dev, test incorporate doc-ids and doc-passage-ids from above

    if unique_passages is None:
        unique_passages = pd.read_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/LongNQ_train_dev_test_passages_wids.tsv", sep="\t", header=0, names=["id","_","text","title","example_ids","splits"])

    data_files = glob.glob("/dccstor/srosent2/generative/appen/final/longNQ/*/*.jsonl")

    dfs = []

    # make questions.tsv for each split
    for file_name in data_files:
        answerable = "answerable"
        if "unanswerable" in file_name:
            answerable = "unanswerable"
        split = "train"
        if "dev" in file_name:
            split = "dev"
        elif "test" in file_name:
            split = "test"
        questions = []
        data = pd.read_json(file_name, lines=True, orient='records', dtype={'id':str})

        for i, row in data.iterrows():
            answers = []
            
            for answer in row['output']:
                if answer['answer'] is not None:
                    answers.append(answer['answer'])
            doc_ids = list(unique_passages[unique_passages['example_ids'].str.contains(row['id'])]['id'].values)
            try:
                questions.append([row['id'],row['input'], '' if doc_ids is None else ",".join(doc_ids), '' if answers == [] else "::".join(answers)])
            except:
                print("error")
        print(file_name)
        pd.DataFrame(questions, columns=["id","question","doc-id-list","answers"]).to_csv(f"/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/question_{split}_{answerable}.tsv","\t", index=False)