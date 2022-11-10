import json
import random
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--in_file", type=str, default="data/NewsQA.train.jsonl",)
parser.add_argument("--out_file_dev", type=str,  default="data/NewsQA.sample.dev.jsonl")
parser.add_argument("--out_file_train", type=str, default="data/NewsQA.sample.train.jsonl")
parser.add_argument("--num", type=int, default=1000, required=False)
parser.add_argument("--seed", type=int, default=42, required=False)

args = parser.parse_args()


def subsample_dataset_random(data_jsonl, sample_num=1000, seed=55):

    id_list = list(range(len(data_jsonl)))
    random.seed(seed)
    random.shuffle(id_list)

    dev_list = id_list[:sample_num]
    train_list = id_list[sample_num:]

    train_data = [data_jsonl[idx] for idx in train_list]
    dev_data = [data_jsonl[idx] for idx in dev_list]

    return train_data, dev_data

def main(args):
    dataset = []
    with open(args.in_file, 'r') as rf:
        for e in rf:
            dataset.append(json.loads(e))

    train_data, dev_data = subsample_dataset_random(dataset, args.num, args.seed)

    with open(args.out_file_train, 'w') as outfile:
        for e in train_data:
            json.dump(e, outfile)
            outfile.write('\n')

    with open(args.out_file_dev, 'w') as outfile:
        for e in dev_data:
            json.dump(e, outfile)
            outfile.write('\n')


if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
