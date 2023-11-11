# in this file, load NQ and preprocess to save offline (adapt and save to disk)
# the preprocessing can be done in batches here so its fast

import datasets
from primeqa.mrc.processors.preprocessors.natural_questions import NaturalQuestionsPreProcessor
from transformers import HfArgumentParser, TrainingArguments, AutoConfig, AutoTokenizer
from glob import glob
import math
from dataclasses import dataclass, field


@dataclass
class NQProcessArguments:
    """
    Arguments pertaining to processing nq.
    """
    
    output_dir: str= field(default='/dccstor/srosent3/primeqa/data/nq_adapted_test/', metadata={"help": "directory to output processed file(s)"}
    )
    split: int= field(default=0, metadata={"help": "split to process, only needed for training"}
    )
    num_splits: int= field(default=15, metadata={"help": "number of splits, only needed for training"}
    )
    preprocessing_num_workers: int= field(default=5, metadata={"help": "number of preprocessing workers"}
    )
    do_train: bool= field(default=False, metadata={"help": "True=train, False=dev"}
    )
    adapt: bool= field(default=False, metadata={"help": "True=process, False=don't process"}
    )
    load: bool= field(default=False, metadata={"help": "True=load, False=don't load"}
    )
    
class NQProcess:
    """
    Class to process the full NQ dataset and adapt to base format offline

    main function is adapt which requires the arguments above
    """

    def __init__(self, args) -> None:
        self._dataset_name = "natural_questions"
        self._dataset_config_name = "default"
        self._beam_runner = "DirectRunner"
        self._preprocessing_num_workers = args.preprocessing_num_workers
        self._output_dir = args.output_dir
        self._split = args.split
        self._do_train = args.do_train
        self._num_splits = args.num_splits

        config = AutoConfig.from_pretrained(
            'xlm-roberta-large'
        )
        tokenizer = AutoTokenizer.from_pretrained(
        'xlm-roberta-large',
        use_fast=True,
        config=config,
        )
        self._preprocessor = NaturalQuestionsPreProcessor(tokenizer=tokenizer, num_workers=self._preprocessing_num_workers, load_from_cache_file=False) 

    def adapt(self):
        training_args = TrainingArguments(
            output_dir=self._output_dir,
            overwrite_output_dir=True,
            do_train=self._do_train,
        )
        # turn off caching
        raw_datasets = datasets.load_dataset(
                            self._dataset_name,
                            self._dataset_config_name,
                            beam_runner=self._beam_runner,
                            revision="main"
                        )
        datasets.disable_caching()

        train_size = len(raw_datasets['train'])
        eval_size = len(raw_datasets['validation'])
        print(f'train size: {train_size}')
        print(f'eval size: {eval_size}')
        
        if self._do_train:
            train_split_size = math.ceil(train_size / self._num_splits)
            start = math.floor(self._split * train_split_size)
            end = start + train_split_size
            if end > train_size:
                end = train_size
            print(f"{start}:{end}")
        else:
            start = 0
            end = eval_size

        if self._do_train:
            with training_args.main_process_first(desc=f"train dataset map pre-processing"):
                train_examples = raw_datasets['train'].select(range(start, end))
                processed_train_dataset = self._preprocessor.adapt_dataset(train_examples,True)
            processed_train_dataset.save_to_disk(self._output_dir + f"/train/subset{start}-{end}")
        else:
            # process val data
            with training_args.main_process_first(desc=f"eval dataset map pre-processing"):
                eval_examples = raw_datasets['validation'].select(range(start, end))
                processed_eval_dataset = self._preprocessor.adapt_dataset(eval_examples,False)
            processed_eval_dataset.save_to_disk(self._output_dir + f"/eval/subset{start}-{end}")

    # to test that it worked correctly
    def load(self):
        train_files = glob(self._output_dir + "/train/*")
        eval_files = glob(self._output_dir + "/eval/*")

        if len(train_files) > 0:
            train_datasets = [] 
            for file in train_files:
                d = datasets.load_from_disk(file)
                print(len(d))
                train_datasets.append(d)
            train_dataset = datasets.concatenate_datasets(train_datasets)
            print(len(train_dataset))
                
        if len(eval_files) > 0:
            eval_datasets = [] 
            for file in eval_files:
                d = datasets.load_from_disk(file)
                print(len(d))
                eval_datasets.append(d)
            eval_dataset = datasets.concatenate_datasets(eval_datasets)
            print(len(eval_dataset))

def main():

    parser = HfArgumentParser(NQProcessArguments)
    args = parser.parse_args_into_dataclasses()[0]
    nq_processor = NQProcess(args)

    if args.adapt:
        nq_processor.adapt()
    if args.load:
        nq_processor.load()

if __name__ == "__main__":
    main()