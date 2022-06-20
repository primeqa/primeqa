import json

def reformat_prediction_file(kilt_file, prediction_file):
    assert prediction_file.endswith(".json") and kilt_data_file.endswith(".jsonl")
    with open(prediction_file, 'r') as f:
        predictions = json.loads(f.read())
    id2pred = {}
    for item in predictions:
        id2pred[item["id"]] = item["prediction_text"] 
    assert len(id2pred) == len(predictions)

    with open(prediction_file[:-5] + "_reformat.json", 'w') as fw:
        with open(kilt_file, 'r') as fr:
            for line in fr:
                result = {}
                data = json.loads(line)
                result["id"] = data["id"]
                result["input"] = data["input"]

                pred = id2pred[data["id"]]
                result["output"] = [{"answer": pred}]
                fw.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    kilt_data_file = "/dccstor/myu/OneQA/oneqa/lfqa_kb/data_models/kilt_eli5/eli5-dev-kilt.jsonl"
    prediction_file = "/dccstor/myu/experiments/eli5_large_beam/eval_predictions.json"
    
    reformat_prediction_file(kilt_data_file, prediction_file)
