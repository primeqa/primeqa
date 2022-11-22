from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from dataclasses import dataclass, field
from examples.mitqa.row_retriever_MITQA import RowRetriever
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
    test_data_path: str = field(
       default='data/hybridqa/test.json', metadata={"help": "Dev data path for training on user's own dataset"}
    )
    row_retriever_model_name_path: str = field(
       default='data/hybridqa/models/rr.bin', metadata={"help": "Row retriever configuration file"}
    )
    row_retriever_prediction_file_path: str = field(
       default='data/hybridqa/predictions/test_pred.json', metadata={"help": "Row retriever configuration file"}
    )
    pos_frac_per_epoch: list = field(
       default=[0.3, 0.3, 0.1, 0.0001, 0.0001], metadata={"help": "Positive fraction per epoch"}
    )
    group_frac_per_epoch: list = field(
       default=[0.0, 0.0, 0.0, 0.0, 0.0], metadata={"help": "Positive fraction per epoch"}
    )
    num_train_epochs_rr: int = field(
       default=2,metadata={"help": "Number of epochs to train the row retriever"}
    )
    save_every_niter_rr: int = field(
       default=100,metadata={"help": "Save model after how many iterations"}
    )
    save_model_path_rr: str = field(
       default='data/hybridqa/models/rr.bin',metadata={"help": "Path to save row retrieval model"}
    )
    
    
    
def run_hybrid_qa():
   print("running hybridqa")
   logger = logging.getLogger(__name__)
   hqa_parser = HfArgumentParser((HybridQAArguments,TrainingArguments))
   hqa_args,t_args, = hqa_parser.parse_args_into_dataclasses()
   train_data_processed = preprocess_data(hqa_args.data_path_root,hqa_args.train_data_path,split="train",test=False)
   dev_processed_data_path = preprocess_data(hqa_args.data_path_root,hqa_args.dev_data_path,split="dev",test=False)
   if hqa_args.test_data_path is not None:
      hqa_args.test = True
      test_processed_data_path = preprocess_data(hqa_args.data_path_root,hqa_args.test_data_path,split="test",test=True)
   rr = RowRetriever(hqa_args,t_args)
   
   
   


    

if __name__ == '__main__':
    run_hybrid_qa()