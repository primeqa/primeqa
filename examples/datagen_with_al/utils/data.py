from collections import defaultdict
from dataclasses import dataclass
from functools import partial
import logging
import re
from typing import Dict, List, Union
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, get_dataset_config_names, DownloadConfig
import configparser
import os

import numpy
from tqdm import tqdm
from primeqa.mrc.processors.preprocessors.squad import SQUADPreprocessor

from primeqa.mrc.trainers.mrc import MRCTrainer

logger = logging.getLogger(__name__)


class EnvInterpolation(configparser.BasicInterpolation):
    """Interpolation which expands environment variables in values."""

    def before_get(self, parser, section, option, value, defaults):
        value = super().before_get(parser, section, option, value, defaults)
        return os.path.expandvars(value)


config = configparser.ConfigParser(inline_comment_prefixes='#', interpolation=EnvInterpolation())
config.read(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets.ini'))
DATASETS = {section: {option: os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', value) if option == 'path' else value for option, value in config.items(section) if option not in config.defaults()} for section in config.sections()}

def dicts_to_feature_dict(dict_iter):
    output = defaultdict(list)
    for _dict in dict_iter:
        for key, value in _dict.items():
            output[key].append(value)
    return output
    

def load_hf_dataset(path: str, **kwargs):
    download_config = DownloadConfig(cache_dir=kwargs['cache_dir']) if 'cache_dir' in kwargs else None
    # print("load hf dataset")
    # input(download_config)
    # check if config is chosen in case dataset has several configs
    # if this is not the case then we load all configs and return them in a dict
    available_configs = get_dataset_config_names(path)
    if len(available_configs) > 1:
        if 'name' in kwargs and kwargs['name'] is not None:
            # config specified
            config_obj = kwargs['name']
            if isinstance(config_obj, str):
                configs = config_obj.strip('[]').split(',')
            else:
                configs = config_obj

            # configs could contain regex -> expand
            available_configs = get_dataset_config_names(path)
            configs_expanded = []
            for config in configs:
                if config[:2] == 'r:':
                    # regex found, match with available configs
                    for available_config in available_configs:
                        if re.search(config[2:], available_config):
                            configs_expanded.append(available_config)
                else:
                    configs_expanded.append(config)
            configs = configs_expanded

            # load specified config if only one is given
            if len(configs) == 1:
                return [load_dataset(path, **dict(kwargs, name=configs[0]), download_config=download_config)]
        else:
            # use all available configs
            # TODO maybe throw error since loading all configs is not obvious to the user and can easily be realized via regex
            configs = get_dataset_config_names(path)
        if not isinstance(configs, (tuple, list, set)):
            configs = [configs]
        return [load_dataset(path, **dict(kwargs, name=config), download_config=download_config) for config in configs]
    else:
        assert len(available_configs) == 1
        return [load_dataset(path, **kwargs, download_config=download_config)]


def get_datasets(paths_or_names: Union[str, List[str]], cache_dir: str = None, concatenate: bool = False, keep_in_memory: bool = False, unpack_fn = None, shuffle_seed: int = 42, num_worker: int = 10):
    """Download data and apply some simple preprocessing"""
        
    def get_dataset_args(path_or_name: str):
        # extract path, split and slice from dataset string
        return re.fullmatch(r"^(.*?)(?:\[(.+)\])?(?::(train|validation|test))??(?::((?:\d*%?)?:?\d*%?))?$", path_or_name).groups()

    def parse_slice(value: str):
        """
        Parses a `slice()` from string, like `start:stop:step`.
        """
        if value:
            parts = value.split(':')
            if len(parts) == 1:
                # slice(stop)
                parts = [None, parts[0]]
            # else: slice(start, stop[, step])
        else:
            # slice()
            parts = []
        return slice(*[int(p) if p else None for p in parts])

    if paths_or_names is None:
        return None
        
    if isinstance(paths_or_names, str):
        paths_or_names = [paths_or_names]
        concatenate = True
    logging.info(f"Loading dataset(s): {', '.join(paths_or_names)}")

    datasets = []

    # collect datasets
    for path_or_name in paths_or_names:
        path_or_name, config, split, slice_ = get_dataset_args(path_or_name)
        # check whether path can be loaded from disk with datasets library
        try:
            # try to load as DatasetDict
            dataset = DatasetDict.load_from_disk(path_or_name, keep_in_memory=keep_in_memory)
            if split is not None:
                dataset = dataset[split]
            if shuffle_seed is not None:
                # most of the time shuffling doesn't hurt and is needed for low-resource experiments where we pick n samples randomly
                # one can disable it by setting `shuffle_seed` to `None`
                dataset = dataset.shuffle(shuffle_seed)
            if slice_ is not None:
                dataset = Dataset.from_dict(dataset[parse_slice(slice_)])
            datasets.append(dataset)
            # continue if path could be loaded
            continue
        except FileNotFoundError:
            pass
        try:
            # try to load as Dataset
            dataset = Dataset.load_from_disk(path_or_name, keep_in_memory=keep_in_memory)
            # enforce unique ids
            assert len(dataset.unique('id')) == len(dataset), "IDs are not unique in the dataset!"
            if shuffle_seed is not None:
                # most of the time shuffling doesn't hurt and is needed for low-resource experiments where we pick n samples randomly
                # one can disable it by setting `shuffle_seed` to `None`
                dataset = dataset.shuffle(shuffle_seed)
            if slice_ is not None:
                dataset = Dataset.from_dict(dataset[parse_slice(slice_)])
            datasets.append(dataset)
            # continue if path could be loaded
            continue
        except FileNotFoundError:
            pass

        # check for custom datasets first, for hf datasets afterwards
        if path_or_name in DATASETS:
            # custom name
            # DATASETS dict may contain split and/or information on how to split data
            # we choose split first (if available in DATASETS dict (loaded from datasets.ini)) because we split only object of type Dataset further
            dataset_kwargs = DATASETS[path_or_name].copy()
            train = dataset_kwargs.pop('train', None)
            validation = dataset_kwargs.pop('validation', None)
            test = dataset_kwargs.pop('test', None)
            shuffle = dataset_kwargs.pop('shuffle', None)
            if config is not None:
                # given config overrides config from .ini file
                dataset_kwargs.update(name=config)
            datasets_loaded = load_hf_dataset(**dataset_kwargs, cache_dir=cache_dir, keep_in_memory=keep_in_memory)
            for dataset in datasets_loaded:
                # split object of type Dataset according to given information
                if train is not None or validation is not None or test is not None:
                    assert isinstance(dataset, Dataset), "train, validation or test split information given but object is not of type Dataset."
                    # we shuffle here for the following reasons:
                    #  1. given a seed in datasets.ini we can guarantee the same splits
                    #  2. shuffle=True in method train_test_split without a seed being set results in the same seed (bug?)
                    if shuffle is not None:
                        dataset = dataset.shuffle(int(shuffle))
                    if test is not None:
                        test_split = float(test)
                        test_datasetdict = dataset.train_test_split(test_split, shuffle=False)
                        train_validation, test = test_datasetdict['train'], test_datasetdict['test']
                    else:
                        test_split = .0
                        train_validation = dataset
                    if train is not None and validation is not None:
                        train_validation_datasetdict = train_validation.train_test_split(float(validation)/(1.0-test_split), float(train)/(1.0-test_split), shuffle=False)
                        train, validation = train_validation_datasetdict['train'], train_validation_datasetdict['test']
                    elif validation is not None:
                        validation_datasetdict = train_validation.train_test_split(float(validation)/(1.0-test_split), shuffle=False)
                        validation = validation_datasetdict['test']
                    elif train is not None:
                        train_datasetdict = train_validation.train_test_split(train_size=float(train)/(1.0-test_split), shuffle=False)
                        train = train_datasetdict['train']
                    dataset_dict = {}
                    if train is not None:
                        dataset_dict['train'] = train
                    if validation is not None:
                        dataset_dict['validation'] = validation
                    if test is not None:
                        dataset_dict['test'] = test
                    if len(dataset_dict) > 1:
                        # create DatasetDict object
                        dataset = DatasetDict(dataset_dict)
                    else:
                        # there is only one dataset in the dict, use it
                        dataset = list(dataset_dict.items())[0]
                # choose split
                if split is not None:
                    assert isinstance(dataset, DatasetDict), "dataset is not a DatasetDict, cannot specify split."
                    if split not in dataset.keys():
                        raise ValueError(f"Split '{split}' does not exist for dataset '{dataset}'. Available splits are '{', '.join(dataset.keys())}'.")
                    dataset = dataset[split]
                if shuffle_seed is not None:
                    # most of the time shuffling doesn't hurt and is needed for low-resource experiments where we pick n samples randomly
                    # can be disabled by setting `shuffle_seed` to `None`
                    dataset = dataset.shuffle(shuffle_seed)
                if slice_ is not None:
                    dataset = Dataset.from_dict(dataset[parse_slice(slice_)])
                datasets.append(dataset)
        else:# path_or_name in list_datasets():
            # name appears in datasets library
            datasets_loaded = load_hf_dataset(path_or_name, name=config, split=split, cache_dir=cache_dir, keep_in_memory=keep_in_memory)
            for dataset in datasets_loaded:
                if shuffle_seed is not None:
                    # most of the time shuffling doesn't hurt and is needed for low-resource experiments where we pick n samples randomly
                    # one can disable it by setting `shuffle_seed` to `None`
                    dataset = dataset.shuffle(shuffle_seed)
                if slice_ is not None:
                    dataset = Dataset.from_dict(dataset[parse_slice(slice_)])
                datasets.append(dataset)
        # else:
        #     raise ValueError(f"{path_or_name} is not a valid dataset name.")
        # print(datasets[-1])
        # print(datasets[-1][0])
        # input()

    # unpack samples if necessary
    if any('questions' in dataset.column_names for dataset in datasets):
        assert unpack_fn is not None, "Please provide a filter mechanism since data is packed"
    for dataset in datasets:
        if 'questions' in dataset.column_names:
            dataset = unpack_fn(dataset)

    if concatenate:
        # concatenate datasets and remove config names
        if len(datasets) == 1:
            # we don't have to concatenate datasets in this case but can return the only one
            return datasets[0]

        # keep only columns all datasets have in common
        feature_set = set(datasets[0].column_names)
        for dataset in datasets:
            feature_set &= set(dataset.column_names)
        logging.info(f"Keeping only columns {', '.join(feature_set)} for concatenation of datasets.")

        for i in range(len(datasets)):
            datasets[i] = datasets[i].remove_columns(set(datasets[i].column_names) - feature_set)
            datasets[i].reset_format()

        # try converting data into same features by re-creating datasets from dicts (sometimes casting doesn't work)
        features = datasets[0].features
        datasets_new = [datasets[0]] if len(datasets[0]) > 0 else []
        for i in range(1, len(datasets)):
            if len(datasets[i]) > 0:
                # the from_dict mehtod will fail if the features don't match (what is ok since one shouldn't concatenate in this case)
                datasets_new.append(Dataset.from_dict(dicts_to_feature_dict((sample for sample in datasets[i])), features=features))
        # cast to same features
        feats = datasets_new[0].features
        datasets_new = [dataset.cast(feats) for dataset in datasets_new]
        return concatenate_datasets(datasets_new)
    return datasets
    

def select_unique(data: Dataset, column: str, seed = None, verbose: bool = False):
    if seed is not None:
        data = data.shuffle(seed=seed)
    unique_set = set()
    def filter_unique(sample):
        if sample[column] not in unique_set:
            unique_set.add(sample[column])
            return True
        if verbose:
            print(f"Value '{sample[column]}' appeared multiple times for column '{column}'")
        return False

    return data.filter(filter_unique, num_proc=1)


def expand_answers(data, separate_answers, force_preprocess: bool = False, keep_in_memory: bool = False, num_processes: int = 1):
    logging.info(f"Expanding answers: {'new instances' if separate_answers else 'simple'}")
    data = data.map(unpack_answers, fn_kwargs={'separate_answers': separate_answers}, batched=True, load_from_cache_file=not force_preprocess, keep_in_memory=keep_in_memory, num_proc=num_processes)
    return data


def unpack_answers(samples: Dict, separate_answers: bool = False):
    if 'answers' not in samples or isinstance(samples['answers'][0]['text'][0], str):
        # samples do not contain an answer or are already unpacked
        return samples

    processed_samples = {k: [] for k in samples}
    keys = samples.keys()
    for values in zip(*[samples[k] for k in keys]):
        if separate_answers:
            # split answer to create new instances
            sample = dict(zip(keys, values))
            for answer_start, text in zip(sample['answers']['answer_start'], sample['answers']['text']):
                for key in keys:
                    if key != 'answers':
                        processed_samples[key].append(sample[key])
                processed_samples['answers'].append({
                    'answer_start': answer_start,
                    'text': text,
                })
        else:
            # simply unpack answer
            for key, value in zip(keys, values):
                if key != 'answers':
                    # copy everything except answer
                    processed_samples[key].append(value)
                else:
                    processed_samples[key].append({
                        'answer_start': [answer_start for list_answer_start in value['answer_start'] for answer_start in list_answer_start],
                        'text': [answer_text for list_answer_text in value['text'] for answer_text in list_answer_text],
                    })
    return processed_samples


def unpack_samples(packed_samples, filter_fn = None):
    samples = {
        'id': [],
        'original_id': [],
        'context': [],
        'question': [],
        'answers': [],
        'score': [],
    }
    for gen_samples in tqdm(packed_samples, desc="Unpacking samples", unit="samples"):
        # gen_samples is a dict
        # apply filter function
        if filter_fn is not None:
            questions, answers, scores = filter_fn(context=gen_samples['context'], questions=gen_samples['questions'], answers=gen_samples['answers'], scores=gen_samples['scores'])
        else:
            questions, answers, scores = gen_samples['questions'], gen_samples['answers'], gen_samples['scores']
        for idx, (question, answer, score) in enumerate(zip(questions, answers, scores)):
            # we append the counter to the original id to use as id for the new sample in order to have unique ids
            samples['id'].append(f"{gen_samples['id']}_{idx}")
            samples['original_id'].append(gen_samples['id'])
            samples['context'].append(gen_samples['context'])
            samples['question'].append(question)
            samples['answers'].append(answer)
            samples['score'].append(score)
    return Dataset.from_dict(samples)


@dataclass
class LMFilter():
    num_keep: int

    def filter_lm_score(self, questions, answers, scores, num_keep, **kwargs):
        # `num_keep` best questions & answers
        indices = numpy.argsort(scores)[:-1 - num_keep:-1]
        return [questions[idx] for idx in indices], [answers[idx] for idx in indices], [scores[idx] for idx in indices]

    def __call__(self, data: Dict, num_keep: int = None):
        # unpack samples and apply filtering
        if num_keep is None:
            num_keep = self.num_keep
        return unpack_samples(data, filter_fn=partial(self.filter_lm_score, num_keep=num_keep))


class RTFilter(MRCTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # load preprocessor
        preprocessor = SQUADPreprocessor(
            stride=128,
            tokenizer=self.tokenizer,
            load_from_cache_file=False,
            max_seq_len=None,
            num_workers=None,
            max_q_char_len=None,
            single_context_multiple_passages=True,
            max_contexts=None,
        )

        def preprocess_data(dataset):
            with self.args.main_process_first(desc="Preprocessing dataset for RT filtering"):
                return preprocessor.process_eval(dataset)

        self.preprocess_fn = preprocess_data


    def __call__(self, data: Dict):
        # unpack samples first
        data = unpack_samples(data)
        # RTFilter needs a context index
        if 'context_idx' not in data.column_names:
            data = data.add_column('context_idx', [0] * len(data))
        # do not use the answer column for preparing the features in order to do inference only
        examples, feats = self.preprocess_fn(data)
        # do prediction
        output = self.predict(feats, examples)
        predictions = output.predictions
        # predictions = {prediction['id']: prediction for prediction in predictions}
        # for each sample there has to be a prediction
        assert len(examples) == len(predictions)
        # filter samples where the prediction matches the generated answer
        examples = examples.filter(lambda x: x['answer_text'][0] == predictions[x['example_id']][0]['span_answer_text'], num_proc=None)
        return examples