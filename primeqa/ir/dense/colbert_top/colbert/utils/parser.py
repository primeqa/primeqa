import os
import copy
import faiss

from argparse import ArgumentParser

import primeqa.ir.dense.colbert_top.colbert.utils.distributed as distributed
from primeqa.ir.dense.colbert_top.colbert.utils.runs import Run
from primeqa.ir.dense.colbert_top.colbert.utils.utils import print_message, timestamp, create_directory


class Arguments():
    def __init__(self, description):
        self.parser = ArgumentParser(description=description)
        self.checks = []

        self.add_argument('--root', dest='root', default='experiments')
        self.add_argument('--experiment', dest='experiment', default='dirty')
        self.add_argument('--run', dest='run', default=Run.name)

        self.add_argument('--local_rank', dest='rank', default=-1, type=int)

    def add_model_parameters(self):
        # Core Arguments
        self.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])
        self.add_argument('--dim', dest='dim', default=128, type=int)
        self.add_argument('--query_maxlen', dest='query_maxlen', default=32, type=int)
        self.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)

        # Filtering-related Arguments
        self.add_argument('--mask-punctuation', dest='mask_punctuation', action='store_true')
        self.add_argument('--no-mask-punctuation', dest='mask_punctuation', action='store_false')
        self.parser.set_defaults(mask_punctuation=True)

        # for handling models in local repository
        self.add_argument('--local_models_repository', dest='local_models_repository', default=None, required=False)

    def add_model_training_parameters(self):
        # NOTE: Providing a checkpoint is one thing, --resume is another, --resume_optimizer is yet another.
        self.add_argument('--resume', dest='resume', default=False, action='store_true')
        self.add_argument('--resume_optimizer', dest='resume_optimizer', default=False, action='store_true')
        self.add_argument('--checkpoint', dest='checkpoint', default=None, required=False)

        self.add_argument('--init_from_lm', dest='init_from_lm', default=None, required=False)
        self.add_argument('--model_type', dest='model_type', default='bert-base-uncased', choices=['bert-base-uncased', 'bert-large-uncased','xlm-roberta-base','xlm-roberta-large', 'tinybert'], required=False)

        self.add_argument('--lr', dest='lr', default=3e-06, type=float)
        self.add_argument('--maxsteps', dest='maxsteps', default=400000, type=int)
        self.add_argument('--bsize', dest='bsize', default=32, type=int)
        self.add_argument('--accumsteps', dest='accumsteps', default=1, type=int)
        self.add_argument('--amp', dest='amp', default=False, action='store_true')
        # adding shuffle option
        self.add_argument('--shuffle_every_epoch', dest='shuffle_every_epoch', default=False, action='store_true')
        # support checkpoint
        self.add_argument('--save_every', dest='save_every', default=None, type=int)
        # TODO: deprecate save_steps and save_epochs
        self.add_argument('--save_steps', dest='save_steps', default=2000, type=int)
        #                  help="Training will save checkpoint at the specified steps. "
        #                       "Overridden by save_epochs.")
        self.add_argument('--save_epochs', dest='save_epochs', default=-1, type=int) # ,
        #                  help="Training will save checkpoint at the specified epochs. Overrides save_steps.")
        self.add_argument('--epochs', dest='epochs', default=10, type=int) #,
        #                  help="Training will end at the earlier of the specified epochs or maxsteps.")

        # used in distillation (Student/Teacher) training
        self.add_argument('--teacher_checkpoint', dest='teacher_checkpoint', default=None, required=False)
        self.add_argument('--student_teacher_temperature', dest='student_teacher_temperature', default=1.0, type=float)
        self.add_argument('--student_teacher_top_loss_weight', dest='student_teacher_top_loss_weight', default=0.5, type=float)
        self.add_argument('--teacher_model_type', dest='teacher_model_type', choices=['bert-base-uncased','bert-large-uncased','roberta-base','roberta-large', 'xlm-roberta-base','xlm-roberta-large','bert-base-multilingual-cased','bert-base-multilingual-uncased'], default=None, required=False )
        self.add_argument('--teacher_doc_maxlen', dest='teacher_doc_maxlen', default=180, type=int)
        self.add_argument('--distill_query_passage_separately', dest='distill_query_passage_separately', default=False, required=False, type=bool)
        self.add_argument('--query_only', dest='query_only', default=False, required=False, type=bool)
        self.add_argument('--loss_function', dest='loss_function', required=False)
        self.add_argument('--query_weight', dest='query_weight', default=0.5, type=float)
        self.add_argument('--use_ib_negatives', dest='use_ib_negatives', default=False, action='store_true')

    def add_model_inference_parameters(self):
        self.add_argument('--checkpoint', dest='checkpoint', required=True)
        self.add_argument('--bsize', dest='bsize', default=128, type=int)
        self.add_argument('--amp', dest='amp', default=False, action='store_true')

    def add_training_input(self):
        self.add_argument('--triples', dest='triples', required=True)
        self.add_argument('--queries', dest='queries', default=None)
        self.add_argument('--collection', dest='collection', default=None)
        # used in distillation (Student/Teacher) training
        self.add_argument('--teacher_triples', dest='teacher_triples', default=None)

        def check_training_input(args):
            assert (args.collection is None) == (args.queries is None), \
                "For training, both (or neither) --collection and --queries must be supplied." \
                "If neither is supplied, the --triples file must contain texts (not PIDs)."

        self.checks.append(check_training_input)

    def add_ranking_input(self):
        self.add_argument('--queries', dest='queries', default=None)
        self.add_argument('--collection', dest='collection', default=None)
        self.add_argument('--qrels', dest='qrels', default=None)
        self.add_argument('--ranks_fn', dest='ranks_fn', required=True)
        self.add_argument('--topK', dest='topK', default=100, type=int)

    def add_reranking_input(self):
        self.add_ranking_input()
        self.add_argument('--topk', dest='topK', required=True)
        self.add_argument('--shortcircuit', dest='shortcircuit', default=False, action='store_true')

    def add_indexing_input(self):
        self.add_argument('--collection', dest='collection', required=True)
        self.add_argument('--index_root', dest='index_root', required=True)
        self.add_argument('--index_name', dest='index_name', required=True)

    def add_compressed_index_input(self):
        self.add_argument('--nbits', dest='nbits', choices=[1, 2, 4], type=int, default=1)
        self.add_argument('--kmeans_niters', type=int, default=4)
        self.add_argument('--num_partitions_max', type=int, default=10000000)

    def add_index_use_input(self):
        self.add_argument('--index_root', dest='index_root', required=True)
        self.add_argument('--index_name', dest='index_name', required=True)
        self.add_argument('--partitions', dest='partitions', default=None, type=int, required=False)
        self.add_argument('--index_path', dest='index_path', default=None, type=str)


    def add_retrieval_input(self):
        self.add_index_use_input()
        self.add_argument('--nprobe', dest='nprobe', default=2, type=int)
        self.add_argument('--ncandidates', dest='ncandidates', type=int, default=8192)
        self.add_argument('--retrieve_only', dest='retrieve_only', default=False, action='store_true')

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def check_arguments(self, args):
        for check in self.checks:
            check(args)

    def parse(self):
        (args, remaining_args) = self.parser.parse_known_args()
        if len(remaining_args):
            print_message(f'arguments not used by ColBERT engine: {remaining_args}')

        self.check_arguments(args)

        args.input_arguments = copy.deepcopy(args)

        args.nranks, args.distributed = distributed.init(args.rank)

        args.nthreads = int(max(os.cpu_count(), faiss.omp_get_max_threads()) * 0.8)
        args.nthreads = max(1, args.nthreads // args.nranks)

        if args.nranks > 1:
            print_message(f"#> Restricting number of threads for FAISS to {args.nthreads} per process",
                          condition=(args.rank == 0))
            faiss.omp_set_num_threads(args.nthreads)

        Run.init(args.rank, args.root, args.experiment, args.run)
        Run._log_args(args)
        Run.info(args.input_arguments.__dict__, '\n')

        return args
