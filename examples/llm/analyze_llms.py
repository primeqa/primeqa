from primeqa.components.reader.prompt import BAMReader, PromptFLANT5Reader
import json
import sys
import os
from dataclasses import dataclass, field, asdict
from rouge import Rouge
import numpy as np
import logging
from transformers import HfArgumentParser
from tqdm import tqdm

# BAM docsL https://bam.res.ibm.com/docs/api-reference

# read in ELI5 dev data and run through LLM service.
rouge = Rouge()
sys.setrecursionlimit(20000)

@dataclass
class LLMAnalyzeArguments:
    """
    Arguments pertaining to processing nq.
    """
    api_key: str = field(
        metadata={"help": "The API key for BAM https://bam.res.ibm.com/"},
        default=None
    )
    model_name: str = field(
        default="google/flan-t5-xxl",
        metadata={"help": "Model"},
    )
    prefix: str = field(
        default="Answer the following question after looking at the text. ",
        metadata={"help": "prefix for the LLM"},
    )
    suffix: str = field(
        default=" Answer: ",
        metadata={"help": "suffix for the LLM"},
    )
    prefix_name: str = field(
        default="default",
        metadata={"help": "The abbreviated name to give the prefix (for naming the directory)"},
    )
    max_new_tokens: int = field(
        default=1024,
        metadata={
            "help": "Maximum length of question and context inputs to the model (in word pieces/bpes)",
        },
    )
    min_new_tokens: int = field(
     default=100,
        metadata={
            "help": "Minimum new tokens that must be generated (in word pieces/bpes)",
        },   
    )
    temperature: float = field(
        default=0, metadata={"help": "The temperature parameter used for generation"}
    )
    top_p: float = field(
        default=1, metadata={"help": "The top_p parameter used for generation"}
    )
    top_k: int = field(
        default=5, metadata={"help": "The top_p parameter used for generation"}
    )
    subset_start: int = field(
        default=-1,
        metadata={'help': 'start offset to process a subset of the dataset'}
    )
    subset_end: int = field(
        default=-1,
        metadata={'help': 'end offset to process a subset of the dataset'}
    )
    output_dir: str= field(
        default='/output/loc/here/jsonl', 
        metadata={"help": "directory to output file(s) in ELI5 format. (jsonl)"}
    )
    input_file: str= field(
        default='/input/loc/here/jsonl', 
        metadata={"help": "directory of input file(s) in ELI5 format. (jsonl)"}
    )
    use_passages: bool = field(
        default=False, metadata={"help": "If true input passages to the LLM (up to 3)"}
    )
    save_passages: bool = field(
        default=False, metadata={"help": "If true save input passages to the LLM to file"}
    )
    n_shot: int = field(
        default = 0,
        metadata={'help': 'number of examples *with* answers to provide to the LLM (0, 1, 2)'}
    )
    reader: str = field(
        default="BAMReader",
        metadata={"help": "The name of the prompt reader to use.",
                  "choices": ["BAMReader", "PromptFLANT5Reader"]
                }
    )

def rougel_score(prediction, ground_truth):
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def metric_max_over_ground_truths(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        if 'answer' in ground_truth:
            score = rougel_score(prediction, ground_truth['answer'])
            scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def load_jsonl(file_name):
    json_lines = []
    with open (file_name, 'r') as f:
        data_lines = f.readlines()
        for line in tqdm(data_lines, desc='Reading every example'):
           json_lines.append(json.loads(line))
    return json_lines

def get_answer(service, instance, args, n_doc=3):

    passages = []
    if args.use_passages:
        i = 0
        for t in instance["passages"]:
            i += 1
            passages.append(t["text"])
            if i >= n_doc:
                break

    r = service.predict([instance["input"]], [passages], **asdict(args))

    metric = metric_max_over_ground_truths(r[0]['text'], instance['output'])
    text_generated = r[0]['text']
    
    return metric, text_generated, passages

def get_examples(n_shot=1):
   return None

def main():

    count = 0
    avg_rougeL = 0
    
    parser = HfArgumentParser(LLMAnalyzeArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    if args.reader == "PromptFLANT5Reader":
        reader = PromptFLANT5Reader
    else:
        reader = BAMReader

    reader = reader(args)
    reader.load(model=args.model_name)

    reference_data = load_jsonl(args.input_file)

    model_dir = args.model_name.replace("/","-") + "/prefix_" + args.prefix_name + "-passages_" + str(args.use_passages) + "-" + \
        str(args.n_shot) + "shot_pktemp-" + str(args.top_p) + "_" + str(args.top_k) + "_" + str(args.temperature) \
        + "-minmaxtok_" + str(args.min_new_tokens) + "_" + str(args.max_new_tokens)

    # generate a unique name for this directory so that we can identify based on the dir

    if not os.path.isdir(args.output_dir):
        logging.error("Missing output directory " + args.output_dir)
        sys.exit(0)
    elif not os.path.isdir(args.output_dir + "/" + model_dir):
        os.makedirs(args.output_dir + "/" + model_dir)

    if args.subset_start == -1:
        args.subset_start = 0
    if args.subset_end == -1 or int(args.subset_end) > len(reference_data):
        args.subset_end = len(reference_data)

    # if os.path.exists(args.output_dir + "/" + model_dir + "/" + 'results-' + str(args.subset_start) + "-" + str(args.subset_end) + '.json'):
    #     logging.error(args.output_dir + "/" + model_dir + "/" + 'results-' + str(args.subset_start) + "-" + str(args.subset_end) + ".json exists and is not empty")
    #     sys.exit(0)
    fp = open(args.output_dir + "/" + model_dir + "/" + 'predictions-' + str(args.subset_start) + "-" + str(args.subset_end) + '.json', 'w')
    fpass = None
    if args.save_passages:
        fpass = open(args.output_dir + "/" + model_dir + "/" + 'passages-' + str(args.subset_start) + "-" + str(args.subset_end) + '.json', 'w')

    selected_data = reference_data[args.subset_start:args.subset_end]

    for instance_id in tqdm(range(0, len(selected_data)), desc='Generating answer for every instance'):
        answer = {}

        rouge_metric, text_generated, passages = get_answer(reader, selected_data[instance_id], args)
        answer['rouge'] = rouge_metric
        answer['text'] = text_generated
        answer['id'] = selected_data[instance_id]['id']
        answer['question'] = selected_data[instance_id]['input']
        if args.save_passages:
            json.dump({'id': answer['id'], 'question': answer['question'], 'passages': passages}, fpass)
            fpass.write("\n")
        json.dump(answer, fp)
        fp.write("\n")
        avg_rougeL += rouge_metric
        count += 1
    fp.close()   
    print("RougeL: " + str(avg_rougeL/count))

if __name__ == '__main__':
   main()
