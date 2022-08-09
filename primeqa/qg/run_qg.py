from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from primeqa.qg.processors.data_loader import QGDataLoader
from dataclasses import dataclass,field
from primeqa.qg.models.qg_model import QGModel
from primeqa.qg.trainers.qg_trainer import QGTrainer
from primeqa.qg.metrics.generation_metrics import rouge_metrics
from primeqa.qg.utils.data_collator import T2TDataCollator
from typing import Optional

import json
import logging
import os
import sys
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
       default='t5-base', metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models" ,
                                    "choices":["t5-base", "t5-small", "google/mt5-small","google/mt5-base"]}
    )
    modality: str = field(
       default='table', metadata={"help": "Whether to generate questions from tables or passages",
                                  "choices":["table", "passage"]}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do we want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name:Optional[str] = field(
        default="wikisql", metadata={"help": "Name of the dataset to train the qg model", 
                                    "choices": ["wikisql", "squad", "squad_v2", "tydiqa"]}
    )
    train_file_path: Optional[str] = field(
        default='train_data.pt',
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        default='valid_data.pt',
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )

@dataclass
class InferenceArguments:
    do_generate: Optional[bool] = field(
        default=False, metadata={"help": "Whether to generate questions"}
    )
    num_questions_per_instance: Optional[int] = field(
        default=5, metadata={"help": "Number of questions to generate per table/passage"}
    )
    max_where_clauses: Optional[int] = field(
        default=1, metadata={"help": "Max number of filters in generated question"}
    )
    data_path:str = field(
        default='examples/qg/sample_table.json', metadata={"help": "Path to JSON file with LIST of tables/passages. Each table \
                              should be a dict with keys 'header' and 'rows', and passages should be str"}
    )
    generate_aggregate: Optional[bool] = field(
        default=False, metadata={"help": "Whether to generate aggregate questions with max, min, sum, etc."}
    )
    gen_output_path: Optional[str] = field(
        default='examples/qg/sample_generation.json', metadata={"help": "path to JSON file where generated questions will be saved"} 
    )

def main(raw_args):
    print(raw_args)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, InferenceArguments))

    if len(raw_args) == 2 and raw_args[1].endswith(".json"):
        model_args, data_args, training_args, inference_args = parser.parse_json_file(json_file=raw_args[1])
    elif len(raw_args) == 1:
        model_args, data_args, training_args = parser.parse_dict(raw_args[0])
    else:
        model_args, data_args, training_args, inference_args = parser.parse_args_into_dataclasses()
    
    # These rguments has to be hardcoded in order for Trainer to work
    training_args.predict_with_generate=True
    training_args.remove_unused_columns = False
    training_args.prediction_loss_only = False
    
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    qg_model = QGModel(model_args.model_name_or_path, modality=model_args.modality)

    if training_args.do_train or training_args.do_eval:
        qgdl = QGDataLoader(
            tokenizer=qg_model.tokenizer,
            dataset_name=data_args.dataset_name,
            input_max_len=data_args.max_len,
            target_max_len=data_args.target_max_len
            )
        
        train_dataset = qgdl.create("train")
        valid_dataset = qgdl.create("validation")

        compute_metrics = rouge_metrics(qg_model.tokenizer)
        

        trainer = QGTrainer(
            model=qg_model.model,
            tokenizer = qg_model.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            data_collator=T2TDataCollator(),
            compute_metrics=compute_metrics
        )

    if training_args.do_train:
        trainer.train(
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        train_result = trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Inference
    if inference_args.do_generate:     
        # There are some arguments to control the type of questions generated such as probability of aggregations, number of where clauses etc. (contd.)
        # These arguments can optionally be provided by the user as inference arguments. 
        # Check out the notebook at primeqa/notebooks/qg/tableqginference.ipynb for more details.    
        
        # aggregation proobablities
        agg_prob = [1., 0., 0., 0., 0., 0.]
        if inference_args.generate_aggregate:
            agg_prob = [0., 0.2, 0.2, 0.2, 0.2, 0.2]
        
        # where clauses
        where_prob = [0.]*5
        for i in range(1, 5):
            if i <= inference_args.max_where_clauses:
                where_prob [i] = 1.
        where_prob = [w/sum(where_prob) for w in where_prob]

        with open(inference_args.data_path) as fp:
            data_list = json.load(fp)
        
        generated_questions = qg_model.generate_questions(
                                data_list,
                                inference_args.num_questions_per_instance,
                                agg_prob,
                                where_prob
                                )
        with open(inference_args.gen_output_path, 'w') as fp:
            json.dump(generated_questions, fp)

    # Evaluation
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(metrics.keys()):
                logger.info("  %s = %s", key, str(metrics[key]))
                writer.write("%s = %s\n" % (key, str(metrics[key])))
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main(sys.argv)