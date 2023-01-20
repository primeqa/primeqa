import traceback
from importlib import import_module
import datasets
import json
import random

def object_reference(reference_as_str: str) -> object:
    """
    Given a fully qualified path to a class reference, return a pointer to the reference.
    This will work with types, functions, methods, and other objects (e.g. dict).

    Args:
        reference_as_str: the fully qualified path (expects the fully qualified path in dot notation,
                          e.g. primeqa.mrc.processors.postprocessors.extractive.ExtractivePostProcessor).

    Returns:
        reference to path given by input

    Raises:
        TypeError: Unable to resolve input path
    """
    def _split_into_class_and_module_name(class_path):
        modules = class_path.split('.')
        if len(modules) > 1:
            return ".".join(modules[:-1]), modules[-1]
        else:
            return class_path, None

    try:
        module_name, object_name = _split_into_class_and_module_name(reference_as_str)
        module_reference = import_module(module_name)
        if object_name is None:
            return module_reference
        else:
            return getattr(module_reference, object_name)
    except Exception as ex:
        traceback.print_exc()  # Shows additional traceback for why imports fail
        raise TypeError(f"Unable to resolve the string {reference_as_str} to a fully qualified class path") from ex


def get_raw_datasets(fof, data_args, task_args, cache_dir, split='train'):
    """
    Load in multiple datasets which are either HuggingFace dataset or local data file.

    Args:
        fof: The file that specifies multiple datasets.
             The fof file needs to be in json, jsonl, or csv format.

             If in csv format, the columns of each line, separated by space, are as follows:
                1. dataset name or path of local data file;
                2. dataset config name or data file format;
                3. sampling rate within range 0.0~1.0, e.g. 0.5 means 50% of the examples are randomly selected and loaded;
                4. preprocessor name.

                If column 2~4 are not given, default values will be used:
                data_args.dataset_config_name for dataset or data_args.data_file_format for data file;
                "1.0" for sampling rate;
                task_args.preprocessor for preprocessor name.

            If in jsonl format, each line is a dictionary consisting of
                {'dataset': dataset_name_or_path_of_data_file,
                 'config': dataset_config_or_data_file_format,
                 'sampling_rate': sampling_rate,
                 'preprocessor': preprocessor_name}
                Fields 'config', 'sampling_rate', and 'preprocessor' are also optional. Default value will be used if
                necessary.

            If in json format, a list of dictionary shown above is expected.

        data_args: data arguments containing dataset_config_name and data_file_format.
        task_args: task arguments containing preprocessor.
        cache_dir: cache dir for downloading datasets.
        split: split of dataset to be loaded. No effect to local data file.

    Returns:
        raw_datasets: list of datasets loaded and sampled.
        preprocessors: list of preprocessor names.

    Raises:
        ValueError: Unable to load datasets or data files.
    """

    try:
        F = open(fof, 'r')
    except:
        raise ValueError(f"Unable to open fof {fof} for reading.")

    datasets_info = []
    if fof.endswith('.json'):
        datasets_info = json.load(F)
    elif fof.endswith('.jsonl'):
        for line in F:
            datasets_info.append(json.loads(line))
    else: # assume csv format and columns separated by space
        for line in F:
            d_info = {}
            columns = line.strip().split()
            if len(columns) == 0:
                continue
            if len(columns) >= 1:
                d_info['dataset'] = columns[0]
            if len(columns) >= 2:
                d_info['config'] = columns[1]
            if len(columns) >= 3:
                d_info['sampling_rate'] = columns[2]
            if len(columns) >= 4:
                d_info['preprocessor'] = columns[3]
            datasets_info.append(d_info)

    if len(datasets_info) == 0:
        raise ValueError(f"No dataset or data file information is found in fof {fof}.")
    for i in range(len(datasets_info)):
        if 'dataset' not in datasets_info[i]:
            raise ValueError(f"Dataset name or path of data file is not found in {i}th record of fof {fof}.")
        if 'config' not in datasets_info[i]:
            # if HuggingFace dataset or self-contained python script for dataset processing
            if datasets_info[i]['dataset'] in datasets.list_datasets() or datasets_info[i]['dataset'].endswith('.py'):
                datasets_info[i]['config'] = data_args.dataset_config_name
            else: # local data file(s)
                datasets_info[i]['config'] = data_args.data_file_format
        if 'sampling_rate' not in datasets_info[i]:
            datasets_info[i]['sampling_rate'] = 1.0
        else:
            try:
                s_rate = float(datasets_info[i]['sampling_rate'])
                assert(s_rate >= 0.0 and s_rate <= 1.0)
                datasets_info[i]['sampling_rate'] = s_rate
            except:
                raise ValueError(f"Sampling rate needs to be a float within 0.0~1.0 if given, but found invalid value "
                                 f"{datasets_info[i]['sampling_rate']} in {i}th record of fof {fof}.")
        if 'preprocessor' not in datasets_info[i]:
            datasets_info[i]['preprocessor'] = task_args.preprocessor

    raw_datasets = []
    preprocessors = []
    for i, d_info in enumerate(datasets_info):
        if d_info['dataset'] in datasets.list_datasets() or d_info['dataset'].endswith('.py'):
            if 'natural_questions' in d_info['dataset']:
                raw_dataset = datasets.load_dataset(
                    d_info['dataset'],
                    d_info['config'],
                    cache_dir=cache_dir,
                    beam_runner="DirectRunner",
                    revision="main",
                    split=split
                )
            else:
                raw_dataset = datasets.load_dataset(
                    d_info['dataset'],
                    d_info['config'],
                    cache_dir=cache_dir,
                    split=split
                )
        else:
            data_files = {split: d_info['dataset']}
            raw_dataset = datasets.load_dataset(
                d_info['config'],
                data_files=data_files,
                cache_dir=cache_dir,
                split=split
            )

        if d_info['sampling_rate'] < 1.0:
            max_samples = int(len(raw_dataset) * d_info['sampling_rate'])
            selected_indices = random.sample(range(len(raw_dataset)), max_samples)
            raw_dataset = raw_dataset.select(selected_indices)

        raw_datasets.append(raw_dataset)
        preprocessors.append(d_info['preprocessor'])

    return raw_datasets, preprocessors


def process_raw_datasets(raw_datasets, preprocessors, training_args, split='train', max_samples=None):
    """
    Process datasets into features.

    Args:
        raw_datasets: list of datasets to be processed.
        preprocessors: list of preprocessors for featurization.
        training_args: training arguments.
        split: split of datasets for logging use.
        max_samples: number of examples of each dataset to be processed.

    Returns:
        example_datasets: list of raw datasets truncated with max_samples.
        feature_datasets: list of feature datasets.
    """

    example_datasets, feature_datasets = [], []
    for i, dataset in enumerate(raw_datasets):
        if max_samples is not None:
            # We will select sample from whole data if argument is specified
            dataset = dataset.select(range(max_samples))
        # Feature Creation
        process_fn = preprocessors[i].process_train if split == 'train' else preprocessors[i].process_eval
        with training_args.main_process_first(desc=f"{split} dataset map pre-processing"):
            examples, features = process_fn(examples=dataset)
            example_datasets.append(examples)
            feature_datasets.append(features)
    return example_datasets, feature_datasets