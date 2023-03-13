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
from primeqa.mrc.metrics.rouge.rouge import ROUGE
import random

# BAM docsL https://bam.res.ibm.com/docs/api-reference

# read in ELI5 dev data and run through LLM service.
# import evaluate
# hf_rouge = evaluate.load('rouge')
# rouge = Rouge()
# sys.setrecursionlimit(20000)

rouge = ROUGE()


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
        default=" Answer",
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
    num_context: int = field(
        default = 3,
        metadata={'help': 'number of passages to provide per question.'}
    )

# def rougel_score(prediction, ground_truth):
#     # no normalization
#     try:
#         scores = hf_rouge.compute(predictions=[prediction], references=[ground_truth])
#         # rouge.get_scores(prediction, ground_truth, avg=True)
#     except ValueError:  # "Hypothesis is empty."
#         return 0.0
#     return scores['rougeLsum']
#     #return scores["rouge-l"]["f"]

# def metric_max_over_ground_truths(prediction, ground_truths):
#     scores_for_ground_truths = []
#     for ground_truth in ground_truths:
#         if 'answer' in ground_truth:
#             score = rougel_score(prediction, ground_truth['answer'])
#             scores_for_ground_truths.append(score)
#     return max(scores_for_ground_truths)

def load_jsonl(file_name):
    json_lines = []
    with open (file_name, 'r') as f:
        data_lines = f.readlines()
        for line in tqdm(data_lines, desc='Reading every example'):
           json_lines.append(json.loads(line))
    return json_lines

def get_answer(service, instance, args, n_doc=3, examples=None):

    passages = []
    if args.use_passages:
        i = 0
        for t in instance["passages"]:
            i += 1
            passages.append(t["text"])
            if i >= n_doc:
                break

    r = service.predict([instance["input"]], [passages], examples=examples, **asdict(args))

    refs = []
    for item in instance['output']:
        if 'answer' in item:
            refs.append(item['answer'])

    hf_metric, kilt_metric = rouge._metric_max_over_ground_truths(r[0]['text'], refs)
    text_generated = r[0]['text']
    
    return hf_metric, kilt_metric, text_generated, passages

def get_examples(reader, data, n_shot=1):
    text = ""
    keys = random.sample(data.keys(), n_shot)
  
    for key in keys:
        example = data[key]
        contexts = []
        psg_cnt = 0
        for passage in example['passages']:
            contexts.append(passage['text'])
            psg_cnt += 1
            if psg_cnt >= 3:
                break
        text += "Example: " + reader.create_prompt(question=example['input'],contexts=contexts,prefix="", suffix=None)
        text += " Answer: " + example['output'][0]['answer']
    return text
def main():

    count = 0
    hf_avg_rougeL = 0
    kilt_avg_rougeL = 0
    
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
        str(args.n_shot) + "shot_pktemp-" + str(args.top_p) + "_" + str(args.top_k) + "_" + str(args.temperature) + "-nctx_" + str(args.num_context) \
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

    # if os.path.exists(args.output_dir + "/" + model_dir + "/" + 'predictions-' + str(args.subset_start) + "-" + str(args.subset_end) + '.json'):
    #      logging.error(args.output_dir + "/" + model_dir + "/" + 'predictions-' + str(args.subset_start) + "-" + str(args.subset_end) + ".json exists and is not empty")
    #      sys.exit(0)
    fp = open(args.output_dir + "/" + model_dir + "/" + 'predictions-' + str(args.subset_start) + "-" + str(args.subset_end) + '.json', 'w')
    fpass = None
    if args.save_passages:
        fpass = open(args.output_dir + "/" + model_dir + "/" + 'passages-' + str(args.subset_start) + "-" + str(args.subset_end) + '.json', 'w')

    selected_data = reference_data[args.subset_start:args.subset_end]

    # for n-shot
    ELI5 = {}
    with open("/dccstor/srosent2/generative/eli5-sample/5-ELI5-train-examples-for-evaluation.jsonl",'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            ELI5[data['id']] = data

    for instance_id in tqdm(range(0, len(selected_data)), desc='Generating answer for every instance'):
        answer = {}
        examples = None

        if args.n_shot != 0:
            examples = get_examples(reader, ELI5, args.n_shot)
        hf_rouge_metric, kilt_rouge_metric, text_generated, passages = get_answer(reader, selected_data[instance_id], args, n_doc=args.num_context, examples=examples)
        answer['hf_rouge'] = hf_rouge_metric
        answer['kilt_rouge'] = kilt_rouge_metric
        answer['text'] = text_generated
        answer['id'] = selected_data[instance_id]['id']
        answer['question'] = selected_data[instance_id]['input']
        if args.save_passages:
            json.dump({'id': answer['id'], 'question': answer['question'], 'passages': passages}, fpass)
            fpass.write("\n")
        json.dump(answer, fp)
        fp.write("\n")
        hf_avg_rougeL += hf_rouge_metric
        kilt_avg_rougeL += kilt_rouge_metric
        count += 1
    fp.close()   
    print("HF RougeL: " + str(hf_avg_rougeL/count))
    print("Kilt RougeL: " + str(kilt_avg_rougeL/count))

if __name__ == '__main__':
   main()
