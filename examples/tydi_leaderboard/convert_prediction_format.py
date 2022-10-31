"""
Convert the PrimeQA output format to the official TyDi leaderboard submission format.
"""
import argparse
import glob
import gzip
import json
import logging
from multiprocessing import Pool
from multiprocessing.dummy import Pool as DummyPool
from typing import Optional, List, Dict, Any, Iterable, Tuple

def _convert_start_and_end_positions_from_chars_to_bytes(example, prediction):
    """
    Converts the target start/end positions from bytes to character offsets.
    """
    if prediction['minimal_answer']['start_byte_offset'] == -1 and prediction['minimal_answer']['end_byte_offset'] == -1:
        return prediction
    context = example['document_plaintext']

    prediction['minimal_answer']['start_byte_offset'] = len(context[:prediction['minimal_answer']['start_byte_offset']].encode('utf-8', errors='replace'))
    prediction['minimal_answer']['end_byte_offset'] = len(context[:prediction['minimal_answer']['end_byte_offset']].encode('utf-8', errors='replace'))

    return prediction

def _get_validated_args(input_args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', type=str, required=True,
                        help='Path to the gzip JSONL data. For multiple files, '
                             'should be a glob pattern (e.g. "/path/to/files-*"')

    parser.add_argument('--predictions_path', type=str, required=True,
                        help='Path to JSON file of predictions.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for converted JSONL predictions')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers for loading gold data')
    parser.add_argument('--verbose', action='store_const', const=logging.DEBUG, default=logging.INFO,
                        help='Enable verbose logging')
    args = parser.parse_args(input_args)
    return args


def _load_orginal_predictions(predictions_path: str) -> Dict[int, Dict[str, Any]]:
    logging.info("Reading predictions from: {}".format(predictions_path))
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    predictions_by_example_id = {int(prediction["example_id"]): prediction for prediction in predictions}
    return predictions_by_example_id


def _load_gold_data_from_file(filename: str) -> List[Dict[str, Any]]:
    logging.info("Reading gold data from: {}".format(filename))
    with gzip.open(filename, 'rb') as f:
        gold_data = list(map(json.loads, f))
    return gold_data


def _load_gold_data(gold_path: str, num_workers: int) -> Dict[int, Dict[str, Any]]:
    gold_paths = glob.glob(gold_path)
    num_actual_workers = min(len(gold_paths), num_workers)
    logging.debug("Using {} pool workers".format(num_actual_workers))
    pool_class = Pool if num_actual_workers > 1 else DummyPool
    with pool_class(num_actual_workers) as p:
        individual_gold_data = p.map(_load_gold_data_from_file, gold_paths)
    gold_data = {example['example_id']: example for examples in individual_gold_data for example in examples}
    return gold_data

def _set_boolean(yes_no):
    if yes_no == 3:
        return "YES"
    if yes_no == 4:
        return "NO"
    return "NONE"

def _convert_nq_prediction_to_tydi(nq_prediction: Dict[str, Any], example: Dict[str, Any]) -> Dict[str, Any]:
    assert int(nq_prediction['example_id']) == example['example_id']
    tydi_prediction = dict(example_id=example['example_id'],
                           minimal_answer_score=nq_prediction['confidence_score'],
                           minimal_answer=dict(start_byte_offset=nq_prediction['start_position'],
                                               end_byte_offset=nq_prediction['end_position']),
                           passage_answer_score=nq_prediction['confidence_score'],
                           yes_no_answer=_set_boolean(nq_prediction['yes_no_answer']),
                           passage_answer_index=nq_prediction['passage_index'],
                           language=example['language'])
    tydi_prediction = _convert_start_and_end_positions_from_chars_to_bytes(example, tydi_prediction)
    return tydi_prediction

def _zip_predictions_and_examples(predictions: Dict[int, Dict[str, Any]], examples: Dict[int, Dict[str, Any]]) -> \
        Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for example_id, example in examples.items():
        yield predictions[example_id], example


def _convert_nq_predictions_to_tydi(predictions: Dict[int, Dict[str, Any]],
                                    gold_data: Dict[int, Dict[str, Any]],
                                    num_workers: int) -> List[Dict[str, Any]]:
    pool_class = Pool if num_workers > 1 else DummyPool
    predictions_and_examples = _zip_predictions_and_examples(predictions, gold_data)
    with pool_class(num_workers) as p:
        tydi_predictions = p.starmap(_convert_nq_prediction_to_tydi, predictions_and_examples)
    return tydi_predictions


def main(input_args: Optional[List[str]] = None):
    args = _get_validated_args(input_args)
    logging.basicConfig(level=args.verbose)
    predictions = _load_orginal_predictions(args.predictions_path)
    gold_data = _load_gold_data(args.gold_path, args.num_workers)
    mismatched_example_ids = set(gold_data.keys()).symmetric_difference(predictions.keys())
    if len(mismatched_example_ids) > 0:
        raise ValueError("Mismatched example ids not in both predictions and gold: {}".format(mismatched_example_ids))

    tydi_predictions = _convert_nq_predictions_to_tydi(predictions, gold_data, args.num_workers)

    logging.info("Writing reformatted predictions in JSONL format to {}".format(args.output_path))
    with open(args.output_path, 'w') as f:
        for tydi_prediction in tydi_predictions:
            f.write(json.dumps(tydi_prediction) + '\n')


if __name__ == '__main__':
    main()