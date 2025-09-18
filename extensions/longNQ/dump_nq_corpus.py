# expand the corpus by adding titles not in it yet
import glob
import pandas as pd
import tqdm

count = 0
count_diffs = 0

question_files = glob.glob("/dccstor/srosent2/primeqa/data/train/nq/*-0*.gz")
# data_files = glob.glob("/dccstor/srosent2/primeqa/data/train/nq-full/nq-train-0*")
# data_files.extend(glob.glob("/dccstor/srosent2/primeqa/data/dev/nq-full/nq-dev-*"))
# print(f"loading {len(data_files)} files")

# dfs = []

# for file_name in data_files:
#     data = pd.read_json(file_name, lines=True, orient='records', dtype={'example_id':str})
#     if 'train' in file_name:
#         data['split'] = 'train'
#     elif 'dev' in file_name:
#         data['split'] = 'dev'
#     else:
#         data['split'] = 'test'
#     dfs.append(data)

# all_data = pd.concat(dfs, ignore_index=True)
# print(len(all_data))

dfs = []

for file_name in question_files:
    data = pd.read_json(file_name, lines=True, orient='records', dtype={'example_id':str})
    if 'train' in file_name:
        data['split'] = 'train'
    elif 'dev' in file_name:
        data['split'] = 'dev'
    else:
        data['split'] = 'test'
    dfs.append(data)

question_data = pd.concat(dfs, ignore_index=True)
print(len(question_data))

data_by_title = {}

for i, row in tqdm.tqdm(question_data.iterrows()):
    count += len(row['passage_answer_candidates'])

    # skip long answers
    if row['annotations'][0]['minimal_answer']['plaintext_start_byte'] == -1:
        continue

    # if row['document_title'] in old_titles or row['document_title'] in data_by_title:
    #     continue
    id = row['document_url'][row['document_url'].rindex("=")+1:]
    title = row['document_title']

    passages = {}

    for candidate in row['passage_answer_candidates']:

        passage_text = bytes(row['document_plaintext'],'utf-8')[candidate['plaintext_start_byte']:candidate['plaintext_end_byte']]
        passages[f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"] = {"passage_text":passage_text,"example_id":[],"split":set([row['split']])}
    if f"{title}_{id}" in data_by_title and data_by_title[f"{title}_{id}"]['passages'] != passages:
       count_diffs += 1
    data_by_title[f"{title}_{id}"] = {'id':set([id]), 'title':title, 'passages': passages} 

print(count_diffs)
unique_passages = {}

for item in data_by_title:
    for passage in data_by_title[item]['passages']:
        passage_text = data_by_title[item]['passages'][passage]['passage_text']        
        unique_passages[f"{list(data_by_title[item]['id'])[0]}_{passage}"] = {'text':data_by_title[item]['passages'][passage]['passage_text'].decode(), 'title': data_by_title[item]['title'], 'id': f"{list(data_by_title[item]['id'])[0]}_{passage}"}

print(f"Num unique passages: {len(unique_passages)}/{count}")
# dump passages to tsv
unique_passages_df = pd.DataFrame.from_dict(unique_passages, orient='index', columns=["id","title","text"])
unique_passages_df.to_csv("/dccstor/srosent2/primeqa/data/corpora/nq_passages.tsv", sep="\t", index=False)
