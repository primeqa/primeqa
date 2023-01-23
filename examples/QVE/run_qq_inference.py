import os
import json
import argparse
from tqdm import tqdm

from primeqa.qg.models.qg_model import QGModel


def handle_args():
    usage='Run QG on examples in PrimeQA format and output generated question with original answer in PrimeQA custom mrc format'	
    parser=argparse.ArgumentParser(usage)
    parser.add_argument('--input_file', type=str, required=True, help="path to jsonl file in PrimeQA custom format for QG") 
    parser.add_argument('--output_file', type = str, required=True, help="output jsonl file")  
    parser.add_argument('--model_name_or_path', type = str, required=False, default="PrimeQA/mt5-base-tydi-question-generator", help="Path to pretrained model or model identifier from huggingface.co/models") 
    parser.add_argument('--num_questions_per_instance', type = int, default=1, required=False, help="Number of questions to generate per passage") 
    parser.add_argument('--bsize', type = int, default=1, required=False, help="batch size") 
    args=parser.parse_args()
    return args

def load_passages(input_path):
    examples = []
    text_list = []
    id_list = []
    answers_list = []
    orig_answers_list = []
    with open(input_path) as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)

            text_list.append(example['context'])
            id_list.append(example['id'])
            orig_answers_list.append(example['answers'])
            answers = list(set(example['answers']['text']))
            answers_list.append(answers)
    return id_list, text_list, answers_list, orig_answers_list, examples

def main():
    args = handle_args()
    
    model_name_or_path = args.model_name_or_path 
    passage_qg_model = QGModel(model_name_or_path, modality='passage')
    
    input_file = args.input_file 
    print("Load examples", input_file)
    id_list, text_list, answers_list, orig_answers_list, examples = load_passages(input_file)
    
    output_file = args.output_file 
    
    mrc_examples = []
    bsize = args.bsize #10
    
    num_questions_per_instance = args.num_questions_per_instance #1
    
    for i in range(0,len(id_list),bsize):

        print(i, "Generating questions", len(mrc_examples))
        generated_examples = passage_qg_model.generate_questions(text_list[i:bsize+i],
                    num_questions_per_instance = num_questions_per_instance,  
                    # agg_prob = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    answers_list = answers_list[i:bsize+i], id_list=id_list[i:bsize+i])
    
        print(i, "Num generated", len(generated_examples))
        for g, example in enumerate(generated_examples):
            
            mrc_example = {
                    'id': example['context_id'],
                    'context':  example['context'],
                    'question': example['question'],
                    'answers': orig_answers_list[i+g] 
                }
            mrc_examples.append(json.dumps(mrc_example))
            
    print(f"Writing {output_file}")
    with open(output_file, 'w') as outfile:
        outfile.writelines([f'{example}\n' for example in mrc_examples])
    print(f"Wrote {output_file}")
        
        
# do main
if __name__=='__main__':
    main()

    
