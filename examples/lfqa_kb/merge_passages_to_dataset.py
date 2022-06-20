import json


def merge(passage_file, data_file, output_file):
    with open (data_file) as fd, open(passage_file) as fp, open(output_file, 'w') as fw:
        for line in fd:
            data = json.loads(line.strip())
            passage_line = fp.readline()
            passage_info = json.loads(passage_line.strip())
            assert data["id"] == passage_info["id"]

            data["passages"] = passage_info["passages"]
            fw.write(json.dumps(data)+'\n')

def main():
    for split in ["train", "dev"]:
        psg_f = f"/dccstor/myu/retrieval/data/KILT/eli5/predictions/dpr/eli5_re2g_{split}.jsonl"
        data_f = f"/dccstor/myu/data/kilt_eli5/eli5-{split}-kilt.json"
        output_f = f"/dccstor/myu/data/kilt_eli5_dpr/eli5-{split}-kilt-dpr.json"
        merge(psg_f, data_f, output_f)

if __name__ == "__main__":
    main()