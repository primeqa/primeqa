import json

# ASQA repo: https://github.com/google-research/language/tree/master/language/asqa
data_fn = "/dccstor/myu/data/asqa/ASQA.json" # the file downloaded from the ASQA repo

with open (data_fn, 'r') as f:
    data = json.loads(f.read())
for split in ["train", "dev", "test"]:
    dataset = data[split]
    with open (f"/dccstor/myu/data/asqa/asqa_{split}.json", 'w') as f:
        for data_id, ex in dataset.items():
            new_example = {}
            new_example["id"] = data_id
            for k,v in ex.items():
                new_example[k] = v
            wikipages = []
            for k,v in ex["wikipages"].items():
                wikipages.append(v)
            new_example["wikipages"] = wikipages
            line = json.dumps(new_example)
            f.write(line+'\n')