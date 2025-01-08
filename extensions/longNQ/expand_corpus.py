# expand the corpus by adding titles not in it yet
import glob
import pandas as pd

count = 0

data_files = glob.glob("/dccstor/srosent2/primeqa/data/train/nq-full/nq-train-21*")
print(data_files)

old_unique_passages = pd.read_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/passages.tsv", sep="\t", header=0, names=["id","text","title"])
old_titles = set(old_unique_passages['title'].to_list())

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

for i, row in all_data.iterrows():
    count += len(row['passage_answer_candidates'])

    if row['document_title'] in old_titles or row['document_title'] in data_by_title:
        continue
    id = row['document_url'][row['document_url'].rindex("=")+1:]
    title = row['document_title']
    
    passages = {}
    index = 0
    start = -1
    end = -1
    for candidate in row['passage_answer_candidates']:

        if start != -1 and \
            candidate['plaintext_start_byte'] >= start and \
                candidate['plaintext_end_byte'] <= end:
            index += 1
            # overlapping candidate
            continue
        passage_text = bytes(row['document_plaintext'],'utf-8')[candidate['plaintext_start_byte']:candidate['plaintext_end_byte']]
        start = candidate['plaintext_start_byte']
        end = candidate['plaintext_end_byte']

        passages[f"{candidate['plaintext_start_byte']}-{candidate['plaintext_end_byte']}"] = {"passage_text":passage_text,"example_id":[],"split":set([row['split']])}
        index += 1

    data_by_title[title] = {'id':set([id]), 'title':title, 'passages': passages} 

unique_passages = {}

for item in data_by_title:
    for passage in data_by_title[item]['passages']:
        passage_text = data_by_title[item]['passages'][passage]['passage_text']
        num_words = len(passage_text.decode().split(" "))
        # discard really short or really long passages
        if data_by_title[item]['passages'][passage]['example_id'] == [] and \
            (num_words < 15 or num_words > 3000):
            continue
        
        unique_passages[f"{list(data_by_title[item]['id'])[0]}_{passage}"] = {'text':data_by_title[item]['passages'][passage]['passage_text'].decode(), 'title': data_by_title[item]['title'], 'id': f"{list(data_by_title[item]['id'])[0]}_{passage}"}

print(f"Num unique passages: {len(unique_passages)}/{count}")
# dump passages to tsv
unique_passages_df = pd.DataFrame.from_dict(unique_passages, orient='index', columns=["id","title","text"])
# unique_passages_df.index.name = 'id'
unique_passages_df = pd.concat([old_unique_passages,unique_passages_df])
unique_passages_df.to_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index_large/passages.tsv", sep="\t", index=False)
