import copy
import itertools
import json
import logging
import math
import os
from collections import defaultdict, namedtuple
from enum import Enum, auto
from itertools import takewhile
from typing import Callable, Dict, Hashable, List, Sequence, Union

import numpy
import torch
from datasets import Dataset, load_metric
from primeqa.mrc.data_models.model_outputs.extractive import ExtractiveQAModelOutput
from primeqa.mrc.trainers.mrc import MRCTrainer
from primeqa.qg.trainers.qg_trainer import GenTrainer
from primeqa.qg.utils.data import find_answer_span
from tqdm import tqdm
from transformers import Trainer

logger = logging.getLogger(__name__)


class AbstractALQueryStrategy:
    """
    An abstract class for scoring samples in the active learning setting. Should be subclassed.
    A strategy instance also holds the queryable dataset hence it should make sure to remove queried samples.
    """

    def __init__(self, trainer_ids: Sequence[Hashable]):
        if type(self) is AbstractALQueryStrategy:
            raise TypeError(
                f"{AbstractALQueryStrategy.__class__.__name__} is not intended to be directly "
                f"instantiated and should be subclassed."
            )

        self.trainer_ids = trainer_ids

    def init(
        self,
        examples: Dataset,
        preprocess_fn_mapping: Dict[Hashable, Callable[[Dataset], Dataset]],
    ):
        # convert examples into features
        self.datasets = [
            preprocess_fn_mapping[trainer_id](examples)[1]
            for trainer_id in self._get_dataset_processing_trainer_ids()
        ]
        # store preprocessing functions in case they are needed later
        self.preprocess_functions = [
            preprocess_fn_mapping[id_] for id_ in self.trainer_ids
        ]

    def query(
        self, num_samples: int, trainers: Sequence[Trainer], sample_id_column: str
    ) -> Union[List[float], numpy.ndarray]:
        raise NotImplementedError()

    def get_trainer_ids(self) -> List[Hashable]:
        return self.trainer_ids

    def _get_dataset_processing_trainer_ids(self) -> List[Hashable]:
        return self.trainer_ids


class AbstractSingleScorePoolBasedALQueryStrategy(AbstractALQueryStrategy):
    """A pool based query strategy using a single score to identify the most useful samples."""

    def init(self, examples: Dataset, preprocess_fn_mapping: Dict):
        super().init(examples, preprocess_fn_mapping)

        # there can be only one dataset for this class which we use as pool
        assert len(self.datasets) == 1
        self.pool = self.datasets[0]

    def query(
        self, num_samples: int, trainers: Sequence[Trainer], sample_id_column: str
    ) -> Union[List[float], numpy.ndarray]:
        scored_instance_indices_sorted = self._score_dataset(self.pool, trainers)
        selected_sample_indices = set()
        # we want to collect `num_samples` samples but one sample might have several instances hence we collect instances until we've got enough samples (or no data left)
        # NOTE we currently select samples according to the min scores of their instances
        # TODO allow to choose between different aggregate functions for all instance scores of sample
        for idx, score in takewhile(
            lambda _: len(selected_sample_indices) < num_samples,
            scored_instance_indices_sorted,
        ):
            selected_sample_indices.add(self.pool[idx][sample_id_column])
        # remove selected samples from pool
        self.pool = self.pool.filter(
            lambda x: x[sample_id_column] not in selected_sample_indices,
            keep_in_memory=True,
        ).flatten_indices()
        return selected_sample_indices

    def _score_dataset(self, dataset: Dataset, trainers: Sequence[Trainer]):
        # evaluate samples from pool for score where higher means more useful

        # some scoring functions will change the mode to enable dropout hence we store current model modes so that we can reset it later
        train_modes = [trainer.model.training for trainer in trainers]

        scores = []
        # order of list `scores` is important therefore we iterate row by row, i.e. row 0, 1, 2, ...
        for sample in tqdm(dataset, desc="Scoring dataset"):
            scores.append(self._score(sample, trainers))

        # reset model modes
        for trainer, train_mode in zip(trainers, train_modes):
            trainer.model.train(train_mode)

        indices_sorted = numpy.argsort(scores)
        return list(zip(indices_sorted.tolist(), numpy.asarray(scores)[indices_sorted]))

    def _score(self, sample: Dataset, trainers: Sequence[Trainer]):
        raise NotImplementedError()


class GenALScorer(AbstractSingleScorePoolBasedALQueryStrategy):
    """This class scores samples based on generated output."""

    class Strategy(Enum):
        SENTENCE_PROBABILITY = "sp"
        SENTENCE_PROBABILITY_DROPOUT = "dsp"
        LEXICAL_SIMILARITY = "ls"
        ROUNDTRIP = "rt"
        SENTENCE_PROBABILITY_DROPOUT_AND_ROUNDTRIP = "dsp+rt"

    def __init__(
        self,
        strategy: Strategy,
        max_gen_length: int,
        special_token_id_map: Dict,
        gen_trainer_id: Hashable,
        rc_trainer_id: Hashable = None,
        num_models: int = 10,
    ):
        trainer_ids = (
            [gen_trainer_id, rc_trainer_id]
            if rc_trainer_id is not None
            else [gen_trainer_id]
        )
        super().__init__(trainer_ids)

        self.strategy = strategy
        self.max_gen_length = max_gen_length
        self.special_token_id_map = special_token_id_map

        ## mode-specific attributes
        # the number of forward passes for the dropout based scoring functions
        self.num_models = num_models
        if strategy == self.Strategy.LEXICAL_SIMILARITY:
            # load metric for lexical comparison
            self.sim_fn = load_metric("meteor")

    def _get_dataset_processing_trainer_ids(self):
        # for preprocessing data we only need the gen_trainer_id which is always the first item in self.trainer_ids
        return [self.trainer_ids[0]]

    def query(self, *args, **kwargs):
        # for `qa2s` discard samples which contain the question in the input (2nd step)
        if "qa2s_step" in self.pool.column_names:
            self.pool = self.pool.filter(
                lambda x: x["qa2s_step"] == 0, keep_in_memory=True
            )

        return super().query(*args, **kwargs)

    def _score(self, sample: Dataset, trainers: Sequence[Trainer]):
        # trainers are in the same order as registered trainer_ids
        gen_trainer: GenTrainer = trainers[0]
        rc_trainer: MRCTrainer = trainers[1] if len(trainers) > 1 else None

        if self.strategy == self.Strategy.SENTENCE_PROBABILITY:
            return self._sentence_probability(sample, gen_trainer)
        elif self.strategy == self.Strategy.SENTENCE_PROBABILITY_DROPOUT:
            return self._sentence_probability_dropout(sample, gen_trainer)
        elif self.strategy == self.Strategy.LEXICAL_SIMILARITY:
            return self._lex_similarity(sample, gen_trainer)
        elif self.strategy in [
            self.Strategy.ROUNDTRIP,
            self.Strategy.SENTENCE_PROBABILITY_DROPOUT_AND_ROUNDTRIP,
        ]:
            score_gen, (
                question_token_ids,
                answer_token_ids,
            ) = self._sentence_probability_dropout(
                sample, gen_trainer, return_generated_sequences=True
            )

            # TODO use GenTrainer? could be useful for generation and extraction of questions and answers (for score and RT filtering)
            # extract question and answer
            question, answer = gen_trainer.tokenizer.convert_tokens_to_string(
                gen_trainer.tokenizer.convert_ids_to_tokens(
                    question_token_ids, skip_special_tokens=True
                )
            ), gen_trainer.tokenizer.convert_tokens_to_string(
                gen_trainer.tokenizer.convert_ids_to_tokens(
                    answer_token_ids, skip_special_tokens=True
                )
            )
            if answer not in sample["context"]:
                # cannot apply RT model since we cannot infer labels therefore return worst score
                score_rc = 0
            else:
                # run prediction and compute score
                char_start = find_answer_span(sample["context"], answer)[0]
                assert char_start >= 0
                rc_sample, rc_features = self.preprocess_functions[1](
                    Dataset.from_dict(
                        {
                            "id": [sample["id"]],
                            "context": [sample["context"]],
                            "question": [question],
                            "answers": [
                                {"answer_start": [char_start], "text": [answer]}
                            ],
                        }
                    )
                )
                metrics = rc_trainer.evaluate(rc_features, rc_sample)
                score_rc = metrics["eval_f1"]

            if self.Strategy.ROUNDTRIP:
                # only return rc score
                return score_rc

            # combine rc and gen score
            score_gen = numpy.square(numpy.exp(numpy.multiply(score_gen, 4)))
            score = numpy.add(score_gen, score_rc)
            return score

    def _generate_output(self, sample: Dataset, trainer: Trainer):
        # TODO this could go to the trainer
        input_ids = torch.tensor([sample["input_ids"]], device=trainer.args.device)
        assert (
            input_ids.size(0) == 1
        ), "We can only handle single instance batches currently"

        if trainer.model.config.is_encoder_decoder:
            assert self.special_token_id_map["bos_token_id"] is not None
        if (
            self.special_token_id_map["bos_token_id_2"] is not None
            and self.special_token_id_map["eos_token_id_2"] is not None
        ):
            # in this case we do two generation steps and the output from the first step has to fit into the model together with the input
            assert (
                input_ids.size(-1) + self.max_gen_length
                <= trainer.tokenizer.model_max_length
            ), (
                input_ids.size(-1),
                self.max_gen_length,
                trainer.tokenizer.model_max_length,
            )
        else:
            assert input_ids.size(-1) < trainer.tokenizer.model_max_length
        # set max_length and ignore max_new_tokens as it counts differently
        generated_sequences = trainer.model.generate(
            input_ids,
            max_length=self.max_gen_length
            if trainer.model.config.is_encoder_decoder
            else input_ids.size(-1) + self.max_gen_length,
            max_new_tokens=None,
            num_return_sequences=1,
            num_beams=10,
            return_dict_in_generate=True,
            output_scores=False,
            pad_token_id=trainer.tokenizer.pad_token_id,
            decoder_start_token_id=self.special_token_id_map["bos_token_id"],
            bos_token_id=self.special_token_id_map["bos_token_id"],
            eos_token_id=self.special_token_id_map["eos_token_id"],
            forced_eos_token_id=None,
        ).sequences

        if trainer.model.config.is_encoder_decoder:
            start_index = 0
        else:
            start_index = input_ids.size(-1)
        if (
            self.special_token_id_map["bos_token_id_2"] is not None
            and self.special_token_id_map["eos_token_id_2"] is not None
        ):
            if trainer.model.config.is_encoder_decoder:
                input_ids = torch.cat(
                    (input_ids, generated_sequences[..., :-1]), dim=-1
                )
            else:
                input_ids = generated_sequences
            assert (
                input_ids.size(-1) <= trainer.tokenizer.model_max_length
            ), f"Input size ({input_ids.size(-1)}) has to be smaller or equal than model max length ({trainer.tokenizer.model_max_length}); generated sequence length in previous step is {generated_sequences.size(-1)} with max generation length set to {self.max_gen_length}"

            generated_sequences = generated_sequences[:, start_index:]
            if not trainer.model.config.is_encoder_decoder:
                start_index = input_ids.size(-1)
            generated_sequences = (
                generated_sequences,
                trainer.model.generate(
                    input_ids,
                    max_length=self.max_gen_length
                    if trainer.model.config.is_encoder_decoder
                    else input_ids.size(-1) + self.max_gen_length,
                    max_new_tokens=None,
                    num_return_sequences=1,
                    num_beams=10,
                    return_dict_in_generate=True,
                    output_scores=False,
                    pad_token_id=trainer.tokenizer.pad_token_id,
                    decoder_start_token_id=self.special_token_id_map["bos_token_id_2"],
                    bos_token_id=self.special_token_id_map["bos_token_id_2"],
                    eos_token_id=self.special_token_id_map["eos_token_id_2"],
                    forced_eos_token_id=None,
                ).sequences[:, start_index:],
            )
        else:
            generated_sequences = (generated_sequences[:, start_index:],)

        return input_ids, generated_sequences

    def _lm_score(self, trainer: Trainer, input_ids, output_ids, lengths=None):
        """This computes the sentence probability using the language model."""
        if lengths is None:
            assert (
                input_ids.size(0) == 1
            ), "For inputs with batch size > 1 you have to pass the `lengths` parameter"
            lengths = input_ids.size(1)

        with torch.no_grad():
            if trainer.model.config.is_encoder_decoder:
                label_ids = output_ids[:, 1:]
                decoder_input_ids = output_ids[:, :-1]
                outputs = trainer.model(
                    input_ids.contiguous(),
                    labels=label_ids.contiguous(),
                    decoder_input_ids=decoder_input_ids.contiguous(),
                )
            else:
                raise NotImplementedError(
                    "LM score computation for non-seq2seq models hasn't been implemented yet."
                )
                outputs = self.model(input_ids, labels=input_ids.clone())
            # outputs[0] is the average negative log likelihood per token
            score = -1.0 * outputs[0].cpu().item() * lengths
        return score

    def _sentence_probability(
        self,
        sample: Dataset,
        trainer: Trainer,
        return_generated_sequences: bool = False,
    ):
        def length_penalty(seq_len, exp):
            return ((5 + seq_len) / 6) ** exp

        input_ids, output_ids = self._generate_output(sample, trainer)

        # we omit the log in the computation here since it's a monotonic function
        score = self._lm_score(trainer, input_ids, output_ids[-1]) / length_penalty(
            input_ids.numel()
            if isinstance(input_ids, torch.Tensor)
            else len(input_ids),
            0.6,
        )
        if return_generated_sequences:
            return score, tuple(_output_ids[0] for _output_ids in output_ids)
        else:
            return score

    def _sentence_probability_dropout(
        self,
        sample: Dataset,
        trainer: Trainer,
        return_generated_sequences: bool = False,
    ):
        # generate output once and use for computing log-likelihood with different models (by activating dropout)
        trainer.model.eval()
        input_ids, output_ids = self._generate_output(sample, trainer)

        # activate dropout
        trainer.model.train()

        # we omit the log in the computation here since it's a monotonic function
        # also we omit the length penalty since this is the D-TP from Fomicheva et al. using averaged scores over samples and tokens
        score = self._lm_score(
            trainer,
            input_ids.expand(self.num_models, -1),
            output_ids[-1].expand(self.num_models, -1),
            lengths=1,
        )
        if return_generated_sequences:
            return score, tuple(_output_ids[0] for _output_ids in output_ids)
        else:
            return score

    def _lex_similarity(
        self,
        sample: Dataset,
        trainer: Trainer,
        return_generated_sequences: bool = False,
    ):
        # generate `num_models` hypotheses with dropout active
        trainer.model.train()
        hyps = [
            trainer.tokenizer.decode(self._generate_output(sample, trainer)[1][-1][0])
            for _ in range(self.num_models)
        ]
        # compute the cartesian product of the list of decoded hypotheses with itself and remove tuples with same hyps (i.e. where i==j)
        hyps_pairwise = list(itertools.product(hyps, repeat=2))
        step = (self.num_models**2 - 1) // (self.num_models - 1)
        for i in range(self.num_models):
            del hyps_pairwise[i * step - i]
        predictions_1, predictions_2 = zip(*hyps_pairwise)
        score = self.sim_fn.compute(
            predictions=predictions_1, references=predictions_2
        )["meteor"]
        if return_generated_sequences:
            raise NotImplementedError(
                "Computation of returned generated sequence hasn't been implemented yet."
            )
            return score, tuple(_output_ids[0] for _output_ids in output_ids)
        else:
            return score


class RCALScorer(AbstractSingleScorePoolBasedALQueryStrategy):
    """This class scores samples based on RC model output using BALD score from http://arxiv.org/abs/1703.02910."""

    def __init__(self, rc_trainer_id: Hashable, num_models: int = 10):
        super().__init__([rc_trainer_id])

        self.num_models = num_models

    def query(
        self, num_samples: int, trainers: Sequence[Trainer], sample_id_column: str
    ) -> Union[List[float], numpy.ndarray]:
        # NOTE current fix for feature column being example_id for RC data
        return super().query(num_samples, trainers, "example_id")

    def _bald(self, sample: Dict, trainer: Trainer):
        # compute bald score (maximize the information gained about the model parameters) with dropout active
        # this is an approximation where we keep start and end probability separate since we have a classification over tokens for start and end probability
        # ideally one would consider the cartesian product of start and end tokens
        trainer.model.train()

        with torch.no_grad():
            # don't use trainer for prediction as we have a dict instead of Dataset
            # by expanding the batch dimension we effectly do `num_models`forward passes
            input_ids = torch.tensor(
                [sample["input_ids"]], device=trainer.args.device
            ).expand(self.num_models, -1)
            token_type_ids = (
                torch.tensor(
                    [sample["token_type_ids"]], device=trainer.args.device
                ).expand(self.num_models, -1)
                if "token_type_ids" in sample
                else None
            )
            attention_mask = torch.tensor(
                [sample["attention_mask"]], device=trainer.args.device
            ).expand(self.num_models, -1)
            output: ExtractiveQAModelOutput = trainer.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            logits = torch.stack((output.start_logits, output.end_logits), dim=1)
            # compute BALD
            mean_over_forward_passes = logits.mean(dim=0)
            score = -(mean_over_forward_passes * mean_over_forward_passes.log()).sum(
                dim=0
            ) + (logits * logits.log()).sum(dim=1).mean(dim=0)
            score = -score.sum().item()

        return score

    def _score(self, sample: Dataset, trainers: Sequence[Trainer]):
        assert len(trainers) == 1
        return self._bald(sample, trainers[0])


TrainerConfig = namedtuple(
    "TrainerConfig",
    (
        "id_",
        "class_",
        "kwargs",
        "preprocess_train_fn",
        "preprocess_eval_fn",
        "init_callback",
    ),
)


class ActiveLearner:
    """
    A class which takes care of active learning scoring samples according to confidence functions.
    Additionally, multiple models can be trained
    """

    def __init__(self, output_dir: str, trainer_configs: List[TrainerConfig]):
        self.trainer_configs: dict[Hashable, TrainerConfig] = {
            trainer_config.id_: trainer_config
            for trainer_config in trainer_configs
            if trainer_config is not None
        }
        self.trainers = {}
        self.metrics = defaultdict(dict)
        # TODO check for best model being loaded at the end? check for popping train_dataset?
        for trainer_id, trainer_config in self.trainer_configs.items():
            if not trainer_config.kwargs["args"].load_best_model_at_end:
                trainer_config.kwargs["args"].load_best_model_at_end = True
                logger.warning(
                    f"`load_best_model_at_end` set to True in order to evaluate correctly after training"
                )
        # dir for storing drawn samples
        self.output_dir = output_dir

    def _create_trainer(
        self,
        examples_train=None,
        iteration: int = None,
        num_total_iterations: int = None,
        skip_instantiated: bool = False,
    ):
        def default_init_fn(class_, kwargs, iteration, num_total_iterations):
            # this default init function updates the run_name as well as the output_dir training arguments for the new iteration if `iteration` is given
            if iteration is not None:
                kwargs["args"] = copy.deepcopy(kwargs["args"])
                # item assignment not supported by TrainingArguments therefore use setattr
                setattr(
                    kwargs["args"],
                    "run_name",
                    f"{kwargs['args'].run_name}_round-{iteration+1}-of-{num_total_iterations}",
                )
                setattr(
                    kwargs["args"],
                    "output_dir",
                    f"{kwargs['args'].output_dir}_round-{iteration+1}-of-{num_total_iterations}",
                )
            return class_(**kwargs)

        if iteration is not None:
            assert num_total_iterations is not None

        # create trainers
        for trainer_config in self.trainer_configs.values():
            if (
                skip_instantiated
                and trainer_config.id_ in self.trainers
                and self.trainers[trainer_config.id_] is not None
            ):
                continue

            if examples_train is not None:
                _, dataset_train_preprocessed = trainer_config.preprocess_train_fn(
                    examples_train
                )
            else:
                dataset_train_preprocessed = None

            # make sure to use correct training data and call init function
            init_kwargs = dict(
                trainer_config.kwargs, train_dataset=dataset_train_preprocessed
            )
            args = (trainer_config.class_, init_kwargs, iteration, num_total_iterations)
            init_fn = (
                trainer_config.init_callback
                if trainer_config.init_callback is not None
                else default_init_fn
            )
            trainer = init_fn(*args)
            # save trainer instance so that we can keep track of them and replace old instances
            self.trainers[trainer_config.id_] = trainer

    def _train(self, iteration: int = None, num_total_iterations: int = None):
        # train models, i.e. call .train() on all trainers
        # this call will re-initialize the model using the given checkpoint
        # TODO make sure to move models between devices to save some GPU memory
        # NOTE this is the bahvior we want to have (i.e. re-train with all samples collected so far)
        for id_, trainer in self.trainers.items():
            # train and record training metrics
            output = trainer.train()
            # input(output.metrics)
            # input(iteration)
            # if iteration is None:
            #     self.metrics[id_] = output.metrics
            # else:
            #     self.metrics[id_][iteration] = output.metrics

    def _evaluate(self, iteration: int):
        # evaluate models, i.e. call .evaluate() on all trainers
        # this call will re-initialize the model using the given checkpoint
        # TODO make sure to move models between devices to save some GPU memory
        # NOTE this is the bahvior we want to have (i.e. re-train with all samples collected so far)
        for id_, trainer in self.trainers.items():
            if self.trainer_configs[id_].kwargs["args"].do_eval:
                # evaluate and record metrics
                metrics = trainer.evaluate()
                if iteration is None:
                    self.metrics[id_].update(metrics)
                else:
                    if iteration not in self.metrics[id_]:
                        self.metrics[id_][iteration] = {}
                    self.metrics[id_][iteration].update(metrics)

    def _get_trainers(self, ids: Sequence):
        return [self.trainers[id_] for id_ in ids]

    def _query(
        self, strategy: AbstractALQueryStrategy, num_samples: int, sample_id_column: str
    ):
        # collect trainers
        trainer_ids = strategy.get_trainer_ids()
        trainers = self._get_trainers(trainer_ids)
        return strategy.query(num_samples, trainers, sample_id_column)

    def _init_strategy(self, strategy: AbstractALQueryStrategy, examples: Dataset):
        strategy.init(
            examples,
            {
                id_: trainer_config.preprocess_eval_fn
                for id_, trainer_config in self.trainer_configs.items()
            },
        )

    def run(
        self,
        examples: Dataset,
        strategy: AbstractALQueryStrategy,
        num_iterations: int,
        num_samples_per_iteration: int,
        example_id_column: str = "id",
        feature_id_column: str = "example_id",
        store_indices_and_scores: bool = True,
        store_samples: bool = True,
    ):
        # this runs active learning using the given data and scoring functions
        # create trainer
        # skip_instantiated=True will make sure that trainers are not re-created if they already exist (e.g. so that they can be used in a second call to run() with the already trained models)
        self._create_trainer(skip_instantiated=True)
        self._init_strategy(strategy, examples)

        # this will contain the sample ids (not the instance/table indices)
        selected_sample_indices = set()
        _num_iterations_requested = num_iterations
        num_iterations = min(
            num_iterations,
            math.ceil(
                len(examples.flatten_indices().unique(example_id_column))
                / num_samples_per_iteration
            ),
        )
        if num_iterations != _num_iterations_requested:
            logger.info(
                f"Requested {_num_iterations_requested} iteration{'s' if _num_iterations_requested != 1 else ''} (with {num_samples_per_iteration} samples each) but available data only allows {num_iterations} iteration{'s' if num_iterations != 1 else ''}."
            )

        for i in range(num_iterations):
            logger.info(f"Running AL round {i + 1}/{num_iterations}")

            queried_sample_ids = self._query(
                strategy, num_samples_per_iteration, feature_id_column
            )
            logger.info(
                "Selected sample ids (%d): %s",
                len(queried_sample_ids),
                sorted(queried_sample_ids),
            )
            selected_sample_indices.update(queried_sample_ids)

            if store_indices_and_scores:
                # store sample ids in output dir
                run_dir = self.output_dir
                sample_ids_filename = os.path.join(
                    run_dir, f"al_samples_round_{i}.json"
                )
                logger.info(f"Storing queried sample ids to {sample_ids_filename}")
                with open(sample_ids_filename, "w") as f:
                    json.dump(
                        {
                            "iteration": i,
                            "num_rounds": num_iterations,
                            "num_new_samples": len(queried_sample_ids),
                            "column": example_id_column,
                            # sort list for debugging because set orders are random
                            "values": sorted(selected_sample_indices),
                        },
                        f,
                    )

            # get selected examples as training data
            examples_train = examples.filter(
                lambda x: x[example_id_column] in selected_sample_indices,
                keep_in_memory=True,
            )

            if store_samples:
                # store samples in output dir
                run_dir = self.output_dir
                samples_dir = os.path.join(run_dir, f"al_samples_round_{i}")
                logger.info(f"Storing queried samples to {samples_dir}")
                examples_train.save_to_disk(samples_dir)

            # create new trainer
            self._create_trainer(
                examples_train=examples_train,
                iteration=i,
                num_total_iterations=num_iterations,
            )

            # train
            self._train(iteration=i, num_total_iterations=num_iterations)
            # evaluate to record metrics
            self._evaluate(iteration=i)

        return self.metrics
