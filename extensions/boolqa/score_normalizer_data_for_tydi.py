import argparse
from datasets import load_dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description='create the data for training the tydi score normalizer')
    parser.add_argument('--output_dir',  
        help='the output directory',
        type=str)  
    args = parser.parse_args()
    return args
    

def main():
    
    args = parse_arguments()
    
    raw_dataset = load_dataset("tydiqa", "primary_task")["train"]
    
    train_splits = raw_dataset.train_test_split(test_size=0.1, seed=42)
    
    train_splits['train'].to_json(args.output_dir + "/tydi_train_train.json")
    
    train_splits['test'].to_json(args.output_dir + "/tydi_train_dev.json")  
 
 
if __name__ == '__main__':
    main()