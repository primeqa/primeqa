import argparse
import os
from datasets import load_dataset
import jsonlines
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='create the data for training the tydi score normalizer')
    parser.add_argument('--train_file',  
        help='the QA train file ',
        type=str)
    parser.add_argument('--eval_file',  
        help='the QA dev file',
        type=str)
    parser.add_argument('--test_file',  
        help='the QA test file',
        type=str)
    parser.add_argument('--search_result_location',  
        help='the output directory with the search result for the train dev and test queries',
        type=str) 
    parser.add_argument('--collection',  
        help='The corpus file with all passages',
        type=str) 
    parser.add_argument('--output_dir',  
        help='the output directory with the train and dev queries',
        type=str)  
    args = parser.parse_args()
    return args
    
def load_ranking_file(filename, max=10):
    qid_to_hits = {}
    with open(filename, 'r') as infile:
        for line in infile:
            qid, pid,rank,score = line.split('\t')
            hit = {
                'rank': int(rank),
                'score': float(score),
                'passage_id': pid,
                'doc_id': pid,
            }
            if qid not in qid_to_hits:
                qid_to_hits[qid] = []
            if len(qid_to_hits[qid]) < max:
                qid_to_hits[qid].append(hit)
    return qid_to_hits

def load_corpus(filename):
    passages = []
    titles = []

    with open(filename, 'r') as infile:
        for line in tqdm(infile):
            id,text,title = line.split('\t')
            passages.append(text)
            titles.append(title)
    return passages,titles

def get_search_results_location_by_split(all_search_results):
    search_result_locations = {"train":[], "dev":[], "test":[]}
    search_result_directories = [name for name in os.listdir(all_search_results) 
                                 if os.path.isdir(os.path.join(all_search_results, name))]
    for name in search_result_directories:
        if name.startswith("train"):
            search_result_locations["train"].append(os.path.join(all_search_results, name))
        elif name.startswith("dev"):
            search_result_locations["dev"].append(os.path.join(all_search_results, name))
        elif name.startswith("test"):
            search_result_locations["test"].append(os.path.join(all_search_results, name))
        else:
            pass
    return search_result_locations
        
def load_data_from_eli5_file(file_name):
    with jsonlines.open(file_name, 'r') as jsonl_f:
        data = [obj for obj in jsonl_f]
    return data
            
def main():
    args = parse_arguments()
    
    raw_data = {}
    raw_data["train"] = load_data_from_eli5_file(args.train_file)
    raw_data["dev"] = load_data_from_eli5_file(args.eval_file)
    if args.test_file:
        raw_data["test"] = load_data_from_eli5_file(args["test_file"])
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    search_result_location_by_split = get_search_results_location_by_split(args.search_result_location)
    
    c_passages,c_titles = load_corpus(args.collection)
        
    for split in raw_data:
        
        dt = raw_data[split]
        
        split_qids_to_hits= {}
        for loc in search_result_location_by_split[split]:
            qids_to_hits = load_ranking_file(os.path.join(loc, "ranked_passages.tsv"))    
            split_qids_to_hits.update(qids_to_hits)

        for ex in dt:
            ex_id = ex['id']
            ex_passages = []
            if ex_id in split_qids_to_hits:
                hits = split_qids_to_hits[ex_id]
                for hit in hits:
                    hit['title'] = c_titles[int(hit['passage_id'])]
                    hit['text'] = c_passages[int(hit['passage_id'])]
                    ex_passages.append(hit)
            ex['passages'] = ex_passages
        
        
        with jsonlines.open(os.path.join(args.output_dir, split+".json"), 'w') as writer:
            writer.write_all(dt)
        
        
if __name__=='__main__':
    main()