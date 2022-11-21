from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from dataclasses import dataclass, field
from processors.preprocessors.preprocess_raw_data import preprocess_data
import logging


@dataclass
class HybridQAArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    data_path_root: str = field(
       default='data/hybridqa/', metadata={"help": "root path to store the preprocessed dataset"}
    )
    train_data_path: str = field(
       default='data/hybridqa/train.json', metadata={"help": "Train data path for training on user's own dataset"}
    )
    dev_data_path: str = field(
       default='data/hybridqa/dev.json', metadata={"help": "Dev data path for training on user's own dataset"}
    )
    
def run_hybrid_qa():
    print("running hybridqa")
    logger = logging.getLogger(__name__)
    hqa_parser = HfArgumentParser((HybridQAArguments,TrainingArguments))
    hqa_args,t_args, = hqa_parser.parse_args_into_dataclasses()
    #train_data_processed = preprocess_data(hqa_args.data_path_root,hqa_args.train_data_path,split="train",test=False)
    dev_data_processed = preprocess_data(hqa_args.data_path_root,hqa_args.dev_data_path,split="dev",test=False)

    

if __name__ == '__main__':
    run_hybrid_qa()