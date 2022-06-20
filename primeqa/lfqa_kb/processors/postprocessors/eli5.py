import datasets
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput

# Post-processing: get multiple references for each answer
def post_processing_function(
    examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput,  tokenizer, data_args
):
    # Decode the predicted tokens.
    preds = outputs.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])} 

    # exampleid2featureid: raw dataset id -> example index ->
    feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
    predictions = {}

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # This is the index of the feature associated to the current example.
        feature_index = feature_per_example[example_index]
        predictions[example["id"]] = decoded_preds[feature_index]

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": [x["answer"] for x in ex[data_args.answer_column] if x["answer"] is not None]} for ex in examples] # muli references


    return EvalPrediction(predictions=formatted_predictions, label_ids=references)
