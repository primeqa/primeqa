import os
import torch

import __main__
from dataclasses import dataclass
from primeqa.ir.dense.colbert_top.colbert.utils.utils import timestamp

from .core_config import DefaultVal


@dataclass
class RunSettings:
    """
        The defaults here have a special status in Run(), which initially calls assign_defaults(),
        so these aren't soft defaults in that specific context.
    """

    overwrite: bool = DefaultVal(False)

    root: str = DefaultVal(os.path.join(os.getcwd(), 'experiments'))
    experiment: str = DefaultVal('default')

    index_root: str = DefaultVal(None)
    name: str = DefaultVal(timestamp(daydir=True))

    rank: int = DefaultVal(0)
    nranks: int = DefaultVal(1)
    amp: bool = DefaultVal(True)

    total_visible_gpus = torch.cuda.device_count()
    gpus: int = DefaultVal(total_visible_gpus)

    @property
    def gpus_(self):
        value = self.gpus

        if isinstance(value, int):
            value = list(range(value))

        if isinstance(value, str):
            value = value.split(',')

        value = list(map(int, value))
        value = sorted(list(set(value)))

        assert all(device_idx in range(0, self.total_visible_gpus) for device_idx in value), value

        return value

    @property
    def index_root_(self):
        return self.index_root or os.path.join(self.root, self.experiment, 'indexes/')

    @property
    def script_name_(self):
        if '__file__' in dir(__main__):
            cwd = os.path.abspath(os.getcwd())
            script_path = os.path.abspath(__main__.__file__)
            root_path = os.path.abspath(self.root)

            if script_path.startswith(cwd):
                script_path = script_path[len(cwd):]

            else:
                try:
                    commonpath = os.path.commonpath([script_path, root_path])
                    script_path = script_path[len(commonpath):]
                except:
                    pass


            if script_path.endswith('bin/pytest'):
                script_path = script_path + '.py'
            assert script_path.endswith('.py'), (script_path, cwd)
            script_name = script_path.replace('/', '.').strip('.')[:-3]

            assert len(script_name) > 0, (script_name, script_path, cwd)

            return script_name

        return None

    @property
    def path_(self):
        return os.path.join(self.root, self.experiment, self.script_name_, self.name) if self.script_name_ is not None else os.path.join(self.root, self.experiment, self.name)

    @property
    def device_(self):
        return self.gpus_[self.rank % self.nranks]


@dataclass
class ResourceSettings:
    checkpoint: str = DefaultVal(None)
    teacher_checkpoint: str = DefaultVal(None)
    triples: str = DefaultVal(None)
    teacher_triples: str = DefaultVal(None)
    collection: str = DefaultVal(None)
    queries: str = DefaultVal(None)
    index_name: str = DefaultVal(None)

@dataclass
class DocSettings:
    dim: int = DefaultVal(128)
    doc_maxlen: int = DefaultVal(180)
    mask_punctuation: bool = DefaultVal(True)


@dataclass
class QuerySettings:
    query_maxlen: int = DefaultVal(32)
    attend_to_mask_tokens : bool = DefaultVal(False)
    interaction: str = DefaultVal('colbert')


@dataclass
class TrainingSettings:
    similarity: str = DefaultVal('cosine')

    bsize: int = DefaultVal(32)

    accumsteps: int = DefaultVal(1)

    lr: float = DefaultVal(3e-06)

    maxsteps: int = DefaultVal(500_000)

    save_every: int = DefaultVal(None)

    resume: bool = DefaultVal(False)
    resume_optimizer: bool = DefaultVal(False)

    ## NEW:
    warmup: int = DefaultVal(None)

    warmup_bert: int = DefaultVal(None)

    relu: bool = DefaultVal(False)

    nway: int = DefaultVal(2)

    use_ib_negatives: bool = DefaultVal(False)

    reranker: bool = DefaultVal(False)

    distillation_alpha: float = DefaultVal(1.0)

    ignore_scores: bool = DefaultVal(False)

    shuffle_every_epoch: bool = DefaultVal(False)

    save_steps: int = DefaultVal(2000)
    save_epochs: int = DefaultVal(-1)
    epochs: int = DefaultVal(10)
    input_arguments: dict = DefaultVal({})
    model_type: str = DefaultVal('bert')
    init_from_lm: str = DefaultVal(None)
    local_models_repository: str = DefaultVal(None)
    ranks_fn: str = DefaultVal(None)
    output_dir: str = DefaultVal(None)
    topK: int = DefaultVal(100)

    # used in distillation (Student/Teacher) training
    student_teacher_temperature: float = DefaultVal(1.0)
    student_teacher_top_loss_weight: float = DefaultVal(0.5)
    teacher_model_type: str = DefaultVal('bert')
    teacher_doc_maxlen: int = DefaultVal(180)
    distill_query_passage_separately: bool = DefaultVal(False)
    query_only: bool = DefaultVal(False)
    loss_function: str = DefaultVal(None)
    query_weight: float = DefaultVal(0.5)

    rng_seed: int = DefaultVal(12345)

@dataclass
class IndexingSettings:
    index_path: str = DefaultVal(None)
    index_location: str = DefaultVal(None)

    nbits: int = DefaultVal(1)

    kmeans_niters: int = DefaultVal(20)

    num_partitions_max: int = DefaultVal(10000000)
    @property
    def index_path_(self):
        return self.index_path or os.path.join(self.index_root_, self.index_name)

@dataclass
class SearchSettings:
    ncells: int = DefaultVal(None)
    centroid_score_threshold: float = DefaultVal(None)
    ndocs: int = DefaultVal(None)
