import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from importlib import import_module
from operator import attrgetter
from typing import Optional, Type
import logging

from transformers import HfArgumentParser
from primeqa.ir.dense.colbert_top.colbert.infra.config.settings import *
from primeqa.ir.sparse.config import BM25Config
from primeqa.ir.sparse.bm25_engine import BM25Engine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
@dataclass
class ProcessArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    engine_type: str = field(
        metadata={"help": "IR engine type"}
    )
    do_train: bool = field(
        default=False, metadata={"help": "Run model training"}
    )
    do_index: bool = field(
        default=False, metadata={"help": "Run data indexing"}
    )
    do_search: bool = field(
        default=False, metadata={"help": "Run search"}
    )

@dataclass
class DPRConfig:
    '''
    to be imported from the DPR implementation
    '''
    pass

def main():

    parser = HfArgumentParser([ProcessArguments, BM25Config])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        process_args, model_args, training_args = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        #process_args, colbert_args, dpr_args, bm25_args = parser.parse_args_into_dataclasses()
        (process_args, bm25_args, remaining_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if process_args.engine_type == 'ColBERT':
        logger.info(f"Running ColBERT")
        from primeqa.ir.dense.colbert_top.colbert.infra import Run, RunConfig
        from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig

        # ColBERT argument parser is use here, to allow additional work done in parse()
        from primeqa.ir.dense.colbert_top.colbert.utils.parser import Arguments

        if hasattr(process_args, 'do_train') and process_args.do_train:
            from primeqa.ir.dense.colbert_top.colbert.trainer import Trainer

            colbert_parser = Arguments(description='ColBERT training')

            colbert_parser.add_model_parameters()
            colbert_parser.add_model_training_parameters()
            colbert_parser.add_training_input()
            args = colbert_parser.parse()

            assert args.bsize % args.accumsteps == 0, ((args.bsize, args.accumsteps),
                                                       "The batch size must be divisible by the number of gradient accumulaParsetion steps.")
            assert args.query_maxlen <= 512
            assert args.doc_maxlen <= 512
            args.lazy = args.collection is not None

            args_dict = vars(args)
            # remove keys not in ColBERTConfig
            args_dict = {key: args_dict[key] for key in args_dict if key not in ['run', 'lazy', 'nthreads', 'distributed', 'input_arguments', 'engine_type', 'do_train']}
            colBERTConfig = ColBERTConfig(**args_dict)

            with Run().context(RunConfig(root=args.root, experiment=args.experiment, nranks=args.nranks, amp=args.amp)):
                trainer = Trainer(args.triples, args.queries, args.collection, colBERTConfig)
                trainer.train(args.checkpoint)
                model_fn = trainer.best_checkpoint_path()
                print('model_fn: ' + model_fn)

        if hasattr(process_args, 'do_index') and process_args.do_index:
            from primeqa.ir.dense.colbert_top.colbert.indexer import Indexer

            colbert_parser = Arguments(description='ColBERT indexing')

            colbert_parser.add_model_parameters()
            colbert_parser.add_model_inference_parameters()
            colbert_parser.add_indexing_input()
            colbert_parser.add_compressed_index_input()
            colbert_parser.add_argument('--nway', dest='nway', default=2, type=int)
            args = colbert_parser.parse()

            args_dict = vars(args)
            # remove keys not in ColBERTConfig
            args_dict = {key: args_dict[key] for key in args_dict if key not in ['run', 'nthreads', 'distributed', 'compression_level', 'input_arguments']}
            # args_dict to ColBERTConfig
            colBERTConfig = ColBERTConfig(**args_dict)

            with Run().context(RunConfig(root=args.root, experiment=args.experiment, nranks=args.nranks, amp=args.amp)):
                indexer = Indexer(args.checkpoint, colBERTConfig)
                indexer.index(name=args.index_name, collection=args.collection, overwrite=True)

        if hasattr(process_args, 'do_search') and process_args.do_search:
            from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher
            parser = Arguments(description='ColBERT search')

            parser.add_model_parameters()
            parser.add_model_inference_parameters()
            parser.add_compressed_index_input()
            parser.add_ranking_input()
            parser.add_retrieval_input()
            args = parser.parse()

            args_dict = vars(args)
            # remove keys not in ColBERTConfig
            args_dict = {key: args_dict[key] for key in args_dict if key not in ['run', 'nthreads', 'distributed', 'compression_level', 'qrels', 'partitions', 'retrieve_only', 'input_arguments']}
            colBERTConfig = ColBERTConfig(**args_dict)

            with Run().context(RunConfig(root=args.root, experiment=args.experiment, nranks=args.nranks, amp=args.amp)):
                searcher = Searcher(args.index_name, checkpoint=args.checkpoint, collection=args.collection, config=colBERTConfig)

                rankings = searcher.search_all(args.queries, args.topK)
                out_fn = args.ranks_fn
                rankings.save(out_fn)

    elif process_args.engine_type == 'DPR':
        pass
    elif process_args.engine_type == 'BM25':
        logger.info(f"Running BM25")

        engine = BM25Engine(bm25_args)
        
        if hasattr(process_args, 'do_index') and process_args.do_index:
            logger.info("Running BM25 indexing")
            engine.do_index()
            logger.info(f"BM25 indexing finished")

        if hasattr(process_args, 'do_search') and process_args.do_search:
            logger.info("Running BM25 search")
            engine.do_search()

            logger.info("BM25 Search finished")
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
