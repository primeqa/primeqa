# open the tydi DEV and TEST files of LongNQ and save all of the passages. Keep unique IDs per passage and do not store duplicates.
# store as <id>\t<text>\t<title>\n

import pandas as pd
import glob
from rouge_score import rouge_scorer
import tqdm


do_corpus = False
do_json_check = True
unique_passages = None
rouge = rouge_scorer.RougeScorer(rouge_types=['rouge1',], split_summaries=False)

if do_corpus:

    print("Create corpus...")


    data_files = glob.glob("/dccstor/srosent2/primeqa/data/*/nq-full/*")

    # ensure priority of ids - test -> dev -> train
    # data_files = glob.glob("/dccstor/srosent2/generative/appen/final/original_tydi/test/*.jsonl")
    # data_files.extend(glob.glob("/dccstor/srosent2/generative/appen/final/original_tydi/dev/*.jsonl"))
    # data_files.extend(glob.glob("/dccstor/srosent2/generative/appen/final/original_tydi/train/*.jsonl"))

    # # old_unique_passages = pd.read_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/LongNQ_train_dev_test_passages_wids.tsv", sep="\t", header=0, names=["id","text","title","example_ids","splits"])
    # old_unique_passages = pd.read_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/passages.tsv", sep="\t", header=0, names=["id","text","title"])
    # # old_unique_passages[['doc_id','pasage_offset']] = old_unique_passages['id'].str.split('_', expand=True)
    # old_doc_ids = set(old_unique_passages['id'].to_list())
    # old_titles = set(old_unique_passages['title'].to_list())
    # old_questions = set(pd.read_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/train/question_train_answerable_nobool.tsv", sep="\t", header=0, dtype={'id':str})['id'].to_list())
    old_doc_ids = None
    old_questions = None

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
    duplicate_gold = 0
    new_unseen_gold = 0

    def compute_length(passage_text):
        passage_len = len(passage_text.decode().split(" "))
        passage_lengths[1] += passage_len
        if passage_len < passage_lengths[0]:
            passage_lengths[0] = passage_len
        if passage_len > passage_lengths[2]:
            passage_lengths[2] = passage_len
        passage_lengths[3]+=1

    for i, row in tqdm.tqdm(all_data.iterrows(), total=len(all_data)):        
        
        count += len(row['passage_answer_candidates'])

        answer_loc = set()
        for ann in row['annotations']:
            if ann['passage_answer']['candidate_index'] != -1:
                answer_loc.add(ann['passage_answer']['candidate_index'])    
        if len(answer_loc) > 1:
            multiple_gold_passages += 1
        num_passages += len(answer_loc)  
        id = row['document_url'][row['document_url'].rindex("=")+1:]
                
        if row['document_title'] in data_by_title:
            duplicates += 1

            # if row['document_url'][row['document_url'].rindex("=")+1:] !=  data_by_title[row['document_title']]['id']:
            #     print(f"New ID for {row['document_title']}")
            if row['document_url'][row['document_url'].rindex("=")+1:] not in data_by_title[row['document_title']]['id']:
                data_by_title[row['document_title']]['id'].append(row['document_url'][row['document_url'].rindex("=")+1:])
            
            for annotation_index in answer_loc:
                candidate = row['passage_answer_candidates'][annotation_index]
                added = False
                passage_text = bytes(row['document_plaintext'],'utf-8')[candidate['plaintext_start_byte']:candidate['plaintext_end_byte']]
                compute_length(passage_text)
                
                # if passage_text not in data_by_title[row['document_title']]['passages'].values():
                # duplicate offsets are super similar so just keep the longer one instead of two versions.
                more_passages += 1
                if f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}" in data_by_title[row['document_title']]['passages']:
                    if data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"]['passage_text'] != passage_text:
                        # rouge_score = rouge.score(data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"]['passage_text'],passage_text)['rouge1'][2]
                        duplicate_gold += 1
                        # print(f"duplicate offsets - similar gold passage ({rouge_score}). keep this one and not the other")
                        # if rouge_score < .98:
                        #     print(data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"]['passage_text'])
                        #     print(passage_text)
                        #     print("----")
                        data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"]['passage_text'] = passage_text
                    # else:
                    #     print("duplicate offsets - same gold passage")
                    data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"]['example_id'].append(row['example_id'])
                    data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"]['document_id'].append(f"{id}_{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}")
                    data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"]['split'].add(row['split'])
                else:
                    # these might be similar to other passages - need to check.
                    matching_passage_id = None
                    for passage_id in data_by_title[row['document_title']]['passages']:
                        rouge_score = rouge.score(data_by_title[row['document_title']]['passages'][passage_id]['passage_text'],passage_text)['rouge1'][0]
                        if rouge_score > .90:
                            matching_passage_id = passage_id
                            break
                    # delete the close match if no answers associated with it.
                    if matching_passage_id != None:
                        if row['split'] == 'train' and old_questions is not None and row['example_id'] not in list(old_questions):
                            data_by_title[row['document_title']]['passages'][matching_passage_id]['example_id'].append(row['example_id'])
                            data_by_title[row['document_title']]['passages'][matching_passage_id]['document_id'].append(f"{id}_{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}")
                            data_by_title[row['document_title']]['passages'][matching_passage_id]['split'].add(row['split'])
                            continue
                        if data_by_title[row['document_title']]['passages'][matching_passage_id]['example_id'] == []:
                            # print(f"adding a new gold passage from a different question -- removing the exact/similar ({rouge_score}) passage that is not gold")
                            del data_by_title[row['document_title']]['passages'][matching_passage_id]
                            data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"] = {"passage_text":passage_text,"example_id":[row['example_id']], "document_id":[f"{id}_{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"],"split":set([row['split']])}
                        else:
                            # print(f"exact/similar ({rouge_score}) passage is gold, keep the longer one.")
                            duplicate_gold += 1
                            if len(passage_text) > len(data_by_title[row['document_title']]['passages'][matching_passage_id]['passage_text']):
                                data_by_title[row['document_title']]['passages'][matching_passage_id]['passage_text'] = passage_text
                                data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"] = data_by_title[row['document_title']]['passages'][matching_passage_id]
                                del data_by_title[row['document_title']]['passages'][matching_passage_id]
                                matching_passage_id = f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"
                            data_by_title[row['document_title']]['passages'][matching_passage_id]['example_id'].append(row['example_id'])
                            data_by_title[row['document_title']]['passages'][matching_passage_id]['document_id'].append(f"{id}_{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}")
                            data_by_title[row['document_title']]['passages'][matching_passage_id]['split'].add(row['split'])
                    else:
                        # print("adding a new gold passage from a different question")
                        new_unseen_gold += 1
                        data_by_title[row['document_title']]['passages'][f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"] = {"passage_text":passage_text,"example_id":[row['example_id']], "document_id":[f"{id}_{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"], "split":set([row['split']])}
        else:
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

                passage_text = bytes(row['document_plaintext'],'utf-8')[candidate['plaintext_start_byte']:candidate['plaintext_end_byte']]
                start = candidate['plaintext_start_byte']
                end = candidate['plaintext_end_byte']

                compute_length(passage_text)
                if index in answer_loc:
                    passages[f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"] = {"passage_text":passage_text,"example_id":[row['example_id']], "document_id":[f"{id}_{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"],"split":set([row['split']])}
                else:
                    passages[f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"] = {"passage_text":passage_text,"example_id":[],"document_id":[],"split":set([row['split']])}
                index += 1

            data_by_title[title] = {'id': [id], 'title':title, 'passages': passages} #, 'answers': answers}

    print(f"multiple gold passages for a single question: {multiple_gold_passages}")
    print(f"duplicate gold passages that are not exact but close: {duplicate_gold}")
    print(f"new unseen gold passages to existing document: {new_unseen_gold}")
    print(f"{duplicates} duplicates including unanswerables. {more_passages} more passages needed to be added.")
    passage_lengths[1] = passage_lengths[1]/passage_lengths[3]
    print(f"min, average, max lengths {passage_lengths}")

    unique_passages = {}
    total_passages = 0
    passages_with_questions = 0
    num_questions = 0
    exact_duplicate_passage_gold = 0

    for item in data_by_title:
        for passage in data_by_title[item]['passages']:
            passage_text = data_by_title[item]['passages'][passage]['passage_text']
            num_words = len(passage_text.decode().split(" "))
            # discard really short or really long passages
            if data_by_title[item]['passages'][passage]['example_id'] == [] and \
                (num_words < 15 or num_words > 3000):
                continue
            if data_by_title[item]['passages'][passage]['example_id'] != []:
                passages_with_questions += 1
                num_questions += len(data_by_title[item]['passages'][passage]['example_id'])
            if data_by_title[item]['id'][0] in unique_passages:
                print('duplicate id')
        
            if data_by_title[item]['passages'][passage]['example_id'] != []:
                if len(data_by_title[item]['passages'][passage]['example_id']) > 1:
                    exact_duplicate_passage_gold += len(data_by_title[item]['passages'][passage]['example_id']) -1
                unique_passages[f"{data_by_title[item]['passages'][passage]['document_id'][0]}"] = {'text':data_by_title[item]['passages'][passage]['passage_text'].decode(), 'title': data_by_title[item]['title'],'example_ids':data_by_title[item]['passages'][passage]['example_id'],'splits':data_by_title[item]['passages'][passage]['split']}
            else:
                unique_passages[f"{data_by_title[item]['id'][0]}_{passage}"] = {'text':data_by_title[item]['passages'][passage]['passage_text'].decode(), 'title': data_by_title[item]['title'],'example_ids':data_by_title[item]['passages'][passage]['example_id'],'splits':data_by_title[item]['passages'][passage]['split']}

    print(f"num duplicate reference passages: {exact_duplicate_passage_gold}")
    print(f"{passages_with_questions} passages with {num_questions} questions and {num_passages} selected passages.")
    print(f"Num unique passages: {len(unique_passages)}/{count}")
    # dump passages to tsv
    unique_passages_df = pd.DataFrame.from_dict(unique_passages, orient='index', columns=["text","title", "example_ids", "splits"])
    unique_passages_df.index.name = 'id'
    # unique_passages_df.drop(columns=["example_ids","splits"]).to_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/passages_wbool.tsv", sep="\t")
    # old_unique_passages = pd.read_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/passages.tsv", sep="\t", header=0)
    # unique_passages_df[~unique_passages_df.index.isin(old_unique_passages['id'])].to_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/passages_just_bool.tsv", sep="\t")
    # unique_passages_df.to_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index_large/LongNQ_train_dev_test_passages_wids.tsv", sep="\t")
    unique_passages_df.drop(columns=["example_ids","splits"]).to_csv("/dccstor/srosent2/primeqa/data/corpora/nq_passages_unique.tsv", sep="\t")
    unique_passages_df.to_csv("/dccstor/srosent2/primeqa/data/corpora/nq_passages_unique_qids.tsv", sep="\t")

if do_json_check:
    import os

    repair_ids = True
    passages = pd.read_csv("/dccstor/srosent2/primeqa/data/corpora/nq_passages_unique_qids.tsv", delimiter="\t", header=0)
    
    # question_files = glob.glob("/dccstor/srosent3/long_nq/retrieval/*/*_answerable.tsv")
    question_files = glob.glob("/proj/srosent3/reinforcement_learning/rag_rlvr/nq_mr2/answerable/nq*.jsonl")

    for question_file in question_files:

        if os.path.exists(question_file.replace("answerable","answerable_adjusted_passages")):
            continue

        no_match = 0
        print(question_file)
        questions = pd.read_json(question_file, lines=True, orient='records', dtype={'conversation_id': str, 'task_id': str})
        contexts = questions.explode('contexts')
        contexts_only = pd.json_normalize(contexts['contexts'])
        contexts['offset'] = contexts.groupby(level=0).cumcount()
        contexts_only.index = contexts.index
        contexts_only = pd.concat([contexts_only, contexts['task_id'], contexts['offset']], axis=1)

        missing_ids = contexts_only[~contexts_only['document_id'].isin(passages['id'])]
        print(f"missing: {len(missing_ids)}")
        print(missing_ids.head(5))

        if repair_ids:
            for i, row in tqdm.tqdm(missing_ids.iterrows(),total=len(missing_ids)):
                question = questions.loc[row.name]
                passage_matches = passages[passages['example_ids'].str.contains(str(row['task_id']))]
                
                if len(passage_matches) == 0:
                    print(f"missing passages: {row['task_id']}")
                    continue
                for j, passage in passage_matches.iterrows():
                    k = max_k = max_rouge = 0
                    score = rouge.score(passage['text'],row['text'])['rouge1'][2]
                    if score > max_rouge:
                        max_rouge = score
                        max_k = k
                    k += 1
                has_target = False
                for target in question['targets']:
                    if target['text'] in passage_matches.iloc[k-1]['text']:
                        has_target = True
                if not has_target:
                    continue
                questions.loc[row.name]['contexts'][row['offset']]['document_id'] = passage_matches.iloc[k-1]['id']
            contexts = questions.explode('contexts')
            contexts_only = pd.json_normalize(contexts['contexts'])
            missing_ids = contexts_only[~contexts_only['document_id'].isin(passages['id'])]
        print("Drop missing IDs")        
        print(f"missing: {len(missing_ids)}")
        questions = questions[~questions.index.isin(missing_ids.index)]
        contexts = questions.explode('contexts')
        contexts_only = pd.json_normalize(contexts['contexts'])
        missing_ids = contexts_only[~contexts_only['document_id'].isin(passages['id'])]
        print(len(missing_ids))
        questions.to_json(question_file.replace("answerable","answerable_adjusted_passages"), lines=True, orient='records')