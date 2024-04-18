import os
import copy

from argparse import ArgumentParser

class Arguments():
    def __init__(self, description):
        self.parser = ArgumentParser(description=description)
        self.add_argument('--local_rank', dest='rank', default=-1, type=int)
        self.add_argument('--rng_seed', dest='rng_seed', default=12345, type=int)

    def add_model_parameters(self):
        # Core Arguments
        self.add_argument('--dim', dest='dim', default=128, type=int)
        self.add_argument('--query_maxlen', dest='query_maxlen', default=32, type=int)
        self.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)
    
    def add_model_training_parameters(self):
        # NOTE: Providing a checkpoint is one thing, --resume is another, --resume_optimizer is yet another.
        self.add_argument('--resume', dest='resume', default=False, action='store_true')
        self.add_argument('--resume_optimizer', dest='resume_optimizer', default=False, action='store_true')
        self.add_argument('--model_name_or_path', dest='model_name_or_path', default=None, required=True)
        self.add_argument('--lr', dest='lr', default=1e-03, type=float)
        self.add_argument('--maxsteps', dest='maxsteps', default=400000, type=int)
        self.add_argument('--max_grad_clip', dest='max_grad_clip', default=2., type=float)
        self.add_argument('--bsize', dest='bsize', default=32, type=int)
        self.add_argument('--k_train', dest='k_train', default=250, type=int)
        self.add_argument('--warmup', dest='warmup', default=None)
        self.add_argument('--accumsteps', dest='accumsteps', default=1, type=int)

        # adding shuffle option
        self.add_argument('--shuffle_every_epoch', dest='shuffle_every_epoch', default=False, action='store_true')
        # support checkpoint
        self.add_argument('--save_every', dest='save_every', default=None, type=int)
        # TODO: deprecate save_steps and save_epochs
        self.add_argument('--save_steps', dest='save_steps', default=2000, type=int)
        self.add_argument('--save_epochs', dest='save_epochs', default=-1, type=int) # ,
        self.add_argument('--epochs', dest='epochs', default=10, type=int) #,

    def add_model_inference_parameters(self):
        #self.add_argument('--model_name_or_path', dest='checkpoint', required=True)
        self.add_argument('--model_name_or_path', dest='model_name_or_path', required=True)
        self.add_argument('--bsize', dest='bsize', default=128, type=int)

    def add_training_input(self):
        self.add_argument('--triples', dest='triples', required=True)
        self.add_argument('--experiment_path', dest='experiment_path', required=True)
        self.add_argument('--nway', dest='nway', default=2, type=int)

    def add_indexing_input(self):
        self.add_argument('--collection', dest='collection', required=True)
        self.add_argument('--index_root', dest='index_root', default=None)
        self.add_argument('--index_name', dest='index_name', required=True)

    def add_compressed_index_input(self):
        self.add_argument('--nbits', dest='nbits', choices=[1, 2, 4], type=int, default=1)
        self.add_argument('--kmeans_niters', type=int, default=4)
        self.add_argument('--num_partitions_max', type=int, default=10000000)

    def add_index_use_input(self):
        self.add_argument('--index_root', dest='index_root', default=None)
        self.add_argument('--index_name', dest='index_name', required=False)
        self.add_argument('--partitions', dest='partitions', default=None, type=int, required=False)
        self.add_argument('--index_location', dest='index_location', default=None, type=str)

    def add_retrieval_input(self):
        self.add_index_use_input()
        self.add_argument('--queries', dest='queries', required=True)
        self.add_argument('--output_dir', dest='output_dir', required=True)
        self.add_argument('--ncells', dest='ncells', default=None, type=int)
        self.add_argument('--centroid_score_threshold', dest='centroid_score_threshold', default=None, type=float)
        self.add_argument('--ndocs', dest='ndocs', default=None, type=int)
        self.add_argument('--topK', dest='topK', default=10, type=int)

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def parse(self):
        (args, remaining_args) = self.parser.parse_known_args()
        if len(remaining_args):
            print(f'arguments not used by XTR engine: {remaining_args}')

        args.input_arguments = copy.deepcopy(args)

        args.nranks = 1

        return args
