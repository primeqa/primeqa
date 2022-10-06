import copy
import itertools
import json
import logging
import math
import os
from collections import namedtuple
from itertools import takewhile
from typing import Dict, List, Sequence, Union

import numpy
import torch
from datasets import Dataset, load_metric
from primeqa.mrc.data_models.model_outputs.extractive import \
    ExtractiveQAModelOutput
from tqdm import tqdm
from transformers import Trainer

logger = logging.getLogger(__name__)


class AbstractALQueryStrategy():
    """
    An abstract class for scoring samples in the active learning setting. Should be subclassed.
    A strategy instance also holds the queryable dataset hence it should make sure to remove queried samples.
    """

    def __init__(self, trainer_ids: Sequence):
        if type(self) is AbstractALQueryStrategy:
            raise TypeError(f"{AbstractALQueryStrategy.__class__.__name__} is not intended to be directly "
                            f"instantiated and should be subclassed.")

        self.trainer_ids = trainer_ids

    def init(self, examples: Dataset, preprocess_fn_mapping: Dict):
        self.datasets = [preprocess_fn_mapping[model_id](examples) for model_id in self.trainer_ids]
    
    def query(self, num_samples: int, trainers: Sequence[Trainer], sample_id_column: str) -> Union[List[float], numpy.ndarray]:
        raise NotImplementedError()

    def get_requested_trainer_ids(self):
        return self.trainer_ids

    def get_num_queryable_samples(self):
        raise NotImplementedError()


class AbstractSingleScorePoolBasedALQueryStrategy(AbstractALQueryStrategy):
    """ A pool based query strategy using a single score to identify the most useful samples. """

    def __init__(self, *args, **kwargs):
        if type(self) is AbstractSingleScorePoolBasedALQueryStrategy:
            raise TypeError(f"{AbstractSingleScorePoolBasedALQueryStrategy.__class__.__name__} is not intended to be directly "
                            f"instantiated and should be subclassed.")
                            
        super().__init__(*args, **kwargs)

        assert len(self.trainer_ids) == 1

    def init(self, examples: Dataset, preprocess_fn_mapping: Dict):
        super().init(examples, preprocess_fn_mapping)

        # there can be only one dataset for this class which we use as pool
        assert len(self.datasets) == 1
        self.pool = self.datasets[0]

    def query(self, num_samples: int, trainers: Sequence[Trainer], sample_id_column: str) -> Union[List[float], numpy.ndarray]:
        assert len(trainers) == 1

        scored_instance_indices_sorted = self._score_dataset(self.pool, trainers[0])
        selected_sample_indices = set()
        # we want to collect `num_samples` samples but one sample might have several instances hence we collect instances until we've got enough samples (or no data left)
        # NOTE we currently select samples according to the min scores of their instances
        # TODO allow to choose between different aggregate functions for all instance scores of sample
        for idx, score in takewhile(lambda _: len(selected_sample_indices) < num_samples, scored_instance_indices_sorted):
            selected_sample_indices.add(self.pool[idx][sample_id_column])
        
        # remove selected samples from pool
        self.pool = self.pool.filter(lambda x: x[sample_id_column] not in selected_sample_indices, keep_in_memory=True)

        return selected_sample_indices

    def _score_dataset(self, dataset: Dataset, trainer: Trainer):
        # store current model mode so that we can reset it later
        train_mode = trainer.model.training
        
        scores = []
        # order of list `scores` is important therefore we iterate row by row, i.e. row 0, 1, 2, ...
        for sample in tqdm(dataset):
            scores.append(self._score(sample, trainer))

        # reset model mode
        trainer.model.train(train_mode)

        indices_sorted = numpy.argsort(scores)
        return list(zip(indices_sorted.tolist(), numpy.asarray(scores)[indices_sorted]))

    def _score(self, sample: Dataset, trainer: Trainer):
        raise NotImplementedError()


class RCALScorer(AbstractSingleScorePoolBasedALQueryStrategy):
    """ This class scores samples based on RC model output using BALD score from http://arxiv.org/abs/1703.02910. """

    def __init__(self, *args, num_models: int = 10, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_models = num_models

    def _bald(self, sample: Dict, trainer: Trainer):
        # compute bald score (maximize the information gained about the model parameters) with dropout active
        # this is an approximation where we keep start and end probability separate since we have a classification over tokens for start and end probability
        # ideally one would consider the cartesian product of start and end tokens
        trainer.model.train()
        
        with torch.no_grad():
            # don't use trainer for prediction as we have a dict instead of Dataset
            # by expanding the batch dimension we effectly do `num_models`forward passes
            input_ids = torch.tensor([sample['input_ids']], device=trainer.args.device).expand(self.num_models, -1)
            token_type_ids = torch.tensor([sample['token_type_ids']], device=trainer.args.device).expand(self.num_models, -1) if 'token_type_ids' in sample else None
            attention_mask = torch.tensor([sample['attention_mask']], device=trainer.args.device).expand(self.num_models, -1)
            output: ExtractiveQAModelOutput = trainer.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            logits = torch.stack((output.start_logits, output.end_logits), dim=1)
            # compute BALD
            mean_over_forward_passes = logits.mean(dim=0)
            score = -(mean_over_forward_passes * mean_over_forward_passes.log()).sum(dim=0) + (logits * logits.log()).sum(dim=1).mean(dim=0)
            score = -score.sum().item()

        return score

    def _score(self, sample: Dataset, trainer: Trainer):
        return self._bald(sample, trainer)


TrainerConfig = namedtuple('TrainerConfig', ('id_', 'class_', 'kwargs', 'preprocess_train_fn', 'preprocess_eval_fn', 'init_callback'))


class ActiveLearner():
    """
    A class which takes care of active learning scoring samples according to confidence functions.
    Additionally, multiple models can be trained 
    """
    
    def __init__(self, output_dir: str, trainer_configs: List[TrainerConfig]):
        self.trainer_configs = {trainer_config.id_: trainer_config for trainer_config in trainer_configs}
        self.trainers = {}
        # TODO check for best model being loaded at the end? check for popping train_dataset?

        # dir for storing drawn samples
        self.output_dir = output_dir
    
    def _create_trainer(self, examples_train = None, iteration: int = None, num_total_iterations: int = None, skip_instantiated: bool = False):
        def default_init_fn(class_, kwargs, iteration, num_total_iterations):
            # this default init function updates the run_name as well as the output_dir training arguments for the new iteration if `iteration` is given
            if iteration is not None:
                kwargs['args'] = copy.deepcopy(kwargs['args'])
                # item assignment not supported by TrainingArguments therefore use stattr
                setattr(kwargs['args'], 'run_name', f"{kwargs['args'].run_name}_round-{iteration+1}-of-{num_total_iterations}")
                setattr(kwargs['args'], 'output_dir', f"{kwargs['args'].output_dir}_round-{iteration+1}-of-{num_total_iterations}")
            return class_(**kwargs)

        if iteration is not None:
            assert num_total_iterations is not None

        # create trainers
        for trainer_config in self.trainer_configs.values():
            if skip_instantiated and trainer_config.id_ in self.trainers and self.trainers[trainer_config.id_] is not None:
                continue

            if examples_train is not None:
                dataset_train_preprocessed = trainer_config.preprocess_train_fn(examples_train)
            else:
                dataset_train_preprocessed = None

            # update kwargs and call init function
            init_kwargs = dict(trainer_config.kwargs, train_dataset=dataset_train_preprocessed)
            args = (trainer_config.class_, init_kwargs, iteration, num_total_iterations)
            init_fn = trainer_config.init_callback if trainer_config.init_callback is not None else default_init_fn
            trainer = init_fn(*args)
            # save trainer instance so that we can keep track of them and replace old instances
            self.trainers[trainer_config.id_] = trainer

    def _train(self, iteration: int = None, num_total_iterations: int = None):
        # train models, i.e. call .train() on all trainers
        # this call will re-initialize the model using the given checkpoint
        # TODO make sure to move models between devices to save some GPU memory
        # NOTE this is the bahvior we want to have (i.e. re-train with all samples collected so far)
        for trainer in self.trainers.values():
            # TODO allow to disable some trainers?
            # TODO always evaluate first, or allow to choose via argument?
            # trainer.evaluate()
            trainer.train()

    def _get_trainers(self, ids: Sequence):
        return [self.trainers[id_] for id_ in ids]
    
    def _get_trainer_configs(self, ids: Sequence):
        return [self.trainer_configs[id_] for id_ in ids]

    def _query(self, strategy: AbstractALQueryStrategy, num_samples: int, sample_id_column: str):
        # collect trainers
        trainer_ids = strategy.get_requested_trainer_ids()
        trainers = self._get_trainers(trainer_ids)
        return strategy.query(num_samples, trainers, sample_id_column)

    def _init_strategy(self, strategy: AbstractALQueryStrategy, examples: Dataset):
        trainer_ids = strategy.get_requested_trainer_ids()
        trainer_configs = self._get_trainer_configs(trainer_ids)
        strategy.init(examples, {id_: trainer_config.preprocess_eval_fn for id_, trainer_config in zip(trainer_ids, trainer_configs)})

    def run(self, examples: Dataset, strategy: AbstractALQueryStrategy, num_iterations: int, num_samples_per_iteration: int, example_id_column: str = 'id', feature_id_column: str = 'example_id', store_indices_and_scores: bool = True, store_samples: bool = True):
        # TODO allow num_samples_per_iteration to be a list with num samples per iteration for all iterations
        # this runs active learning using the given data and scoring functions
        # NOTE data contains twice as many instances per samples in case of qa2s but they have the same sample id and should return the same or a similar similarity score (but they have differently prepared inputs!)
        # create trainer
        # skip_instantiated=True will make sure that trainers are not re-created if they already exist (e.g. so that they can be used in a second call to run() with the already trained models)
        self._create_trainer(skip_instantiated=True)
        self._init_strategy(strategy, examples)

        # this will contain the sample ids (not the instance/table indices)
        selected_sample_indices = set()
        num_iterations = min(num_iterations, math.ceil(len(examples.flatten_indices().unique(example_id_column))/num_samples_per_iteration))
        for i in range(num_iterations):
            logging.info(f"Running AL round {i + 1}/{num_iterations}")
            
            queried_sample_ids = self._query(strategy, num_samples_per_iteration, feature_id_column)
            logger.info("Selected sample ids (%d): %s", len(queried_sample_ids), sorted(queried_sample_ids))
            selected_sample_indices.update(queried_sample_ids)

            if store_indices_and_scores:
                # store sample ids in output dir
                run_dir = self.output_dir
                sample_ids_filename = os.path.join(run_dir, f"al_samples_round_{i}.json")
                logger.info(f"Storing queried sample ids to {sample_ids_filename}")
                with open(sample_ids_filename, 'w') as f:
                    json.dump({
                        'iteration': i,
                        'num_rounds': num_iterations,
                        'num_new_samples': len(queried_sample_ids),
                        'column': example_id_column,
                        # sort list for debugging because set orders are random
                        'values': sorted(selected_sample_indices),
                        }, f)

            # get selected examples as training data
            examples_train = examples.filter(lambda x: x[example_id_column] in selected_sample_indices, keep_in_memory=True)
            
            if store_samples:
                # store samples in output dir
                run_dir = self.output_dir
                samples_dir = os.path.join(run_dir, f"al_samples_round_{i}")
                logger.info(f"Storing queried samples to {samples_dir}")
                examples_train.save_to_disk(samples_dir)

            # create new trainer
            self._create_trainer(examples_train=examples_train, iteration=i, num_total_iterations=num_iterations)

            # train
            self._train(iteration=i, num_total_iterations=num_iterations)

        # TODO return metrics, maybe from train and accumulated over iterations?
