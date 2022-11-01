import argparse
from datasets import load_dataset
import os
import csv

def parse_arguments():
    parser = argparse.ArgumentParser(description='create the data for training the tydi score normalizer')
    parser.add_argument('--train_file',  
        help='the QA train file ',
        type=str)
    parser.add_argument('--eval_file',  
        help='the QA dev file',
        type=str)
    parser.add_argument('--queries_per_file',  
        help='the number of queries in each output file',
        type=int)
    parser.add_argument('--output_dir',  
        help='the output directory with the train and dev queries',
        type=str)  
    args = parser.parse_args()
    return args
    
def main():
    args = parse_arguments()
    
    raw_datasets = {}
    raw_datasets["train"] = load_dataset("json", data_files={"train": args.train_file}, split="train")
    raw_datasets["validation"] = load_dataset("json", data_files={"validation": args.eval_file}, split="validation")
    
     # if the output directory does not exist, create a new directory 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    train_ids = raw_datasets['train']['id']
    train_questions = raw_datasets['train']['input']
    
    writer = None
    tsv_writer = None
    
    n = 0
    for t_id, t_question in zip(train_ids,train_questions):
        if n % args.queries_per_file == 0:
            if n > 0:
                writer.close()
            file_id = n // args.queries_per_file
            writer = open(os.path.join(args.output_dir, f'train_{file_id}'), 'w', encoding='utf-8')
            tsv_writer = csv.writer(writer, delimiter="\t")
        t_question = t_question.replace("\n", "")
        tsv_writer.writerow([t_id, t_question])
        n += 1
    writer.close()
    
    dev_ids = raw_datasets['validation']['id']
    dev_questions = raw_datasets['validation']['input']
    
    n = 0
    for d_id, d_question in zip(dev_ids, dev_questions):
        if n % args.queries_per_file == 0:
            if n > 0:
                writer.close()
            file_id = n // args.queries_per_file
            writer = open(os.path.join(args.output_dir, f'dev_{file_id}'), 'w', encoding='utf-8')
            tsv_writer = csv.writer(writer, delimiter="\t")
        tsv_writer.writerow([d_id, d_question])
        n += 1
    writer.close()
    
        
if __name__ == '__main__':
    main()