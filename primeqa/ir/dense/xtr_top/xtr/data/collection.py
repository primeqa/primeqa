
# Could be .tsv or .json. The latter always allows more customization via optional parameters.
# I think it could be worth doing some kind of parallel reads too, if the file exceeds 1 GiBs.
# Just need to use a datastructure that shares things across processes without too much pickling.
# I think multiprocessing.Manager can do that!

import os
import csv
import json
import itertools
from tqdm import tqdm

from primeqa.ir.dense.colbert_top.colbert.infra.run import Run


class Collection:
    def __init__(self, path=None, data=None):
        self.path = path
        if data is not None:
            self.data = data
        else:
            self.data, self.pids = self._load_file(path)

    def __iter__(self):
        # TODO: If __data isn't there, stream from disk!
        return self.data.__iter__()

    def __getitem__(self, item):
        # TODO: Load from disk the first time this is called. Unless self.data is already not None.
        return self.data[item]

    def __len__(self):
        # TODO: Load here too. Basically, let's make data a property function and, on first call, either load or get __data.
        return len(self.data)

    def _load_file(self, path):
        self.path = path
        return self._load_tsv(path) if path.endswith('.tsv') else self._load_jsonl(path)

    def _load_tsv(self, path):
        print("#> Loading collection...")

        pids = []
        collection = []

        with open(path) as f:
            csv_reader = csv.DictReader(f, fieldnames=["pid", "passage", "title"], delimiter="\t")
            for line_idx, row in enumerate(csv_reader):
                if line_idx % (1000*1000) == 0:
                    print(f'{line_idx // 1000 // 1000}M', end=' ', flush=True)

                pid = row["pid"]
                if pid == "id": continue

                passage = row["passage"]
                if row["title"] is not None:
                    passage = row["title"] + ' | ' + passage
                collection.append(passage)
                pids.append(pid)

        return collection, pids

    def _load_jsonl(self, path, chunk_size=256):
        print("#> Loading collection...")

        pids = []
        collection = []

        with open(path) as f:
            for line_idx, row in tqdm(enumerate(f), desc="Reading Collection"):
                row = json.loads(row)
                passage = row["text"].replace('<s>', '<S>').replace('</s>','</S>')
                pid = row["_id"]
                if row["title"] is not None:
                    title = f"{row['title']}"

                if not passage.strip():continue
                if chunk_size is not None:
                    passage = f"{title} {passage}"
                    tokens = passage.split()[:350] #[:2048]
                    chunk_size = len(tokens)
                    for chunk in range(0, len(tokens), chunk_size):
                        p_text = " ".join(tokens[chunk:chunk+chunk_size])
                        #chunk_text = f"{title} {p_text}"
                        #collection.append(chunk_text)
                        collection.append(p_text)
                        pids.append(pid)
                else:
                    collection.append(passage)
                    pids.append(pid)
        return collection, pids

    def provenance(self):
        return self.path
    
    def toDict(self):
        return {'provenance': self.provenance()}

    def save(self, new_path):
        assert new_path.endswith('.tsv'), "TODO: Support .json[l] too."
        assert not os.path.exists(new_path), new_path

        with Run().open(new_path, 'w') as f:
            # TODO: expects content to always be a string here; no separate title!
            for pid, content in enumerate(self.data):
                content = f'{pid}\t{content}\n'
                f.write(content)
            
            return f.name

    def enumerate(self, rank):
        for _, offset, passages in self.enumerate_batches(rank=rank):
            for idx, passage in enumerate(passages):
                yield (offset + idx, passage)

    def enumerate_batches(self, rank, chunksize=None):
        assert rank is not None, "TODO: Add support for the rank=None case."

        chunksize = chunksize or self.get_chunksize()

        offset = 0
        iterator = iter(self)

        for chunk_idx, owner in enumerate(itertools.cycle(range(Run().nranks))):
            L = [line for _, line in zip(range(chunksize), iterator)]
            

            if len(L) > 0 and owner == rank:
                yield (chunk_idx, offset, L)

            offset += len(L)

            if len(L) < chunksize:
                return
    
    def get_chunksize(self):
        return min(25_000, 1 + len(self) // Run().nranks)  # 25k is great, 10k allows things to reside on GPU??

    @classmethod
    def cast(cls, obj):
        if type(obj) is str:
            return cls(path=obj)

        if type(obj) is list:
            return cls(data=obj)

        if type(obj) is cls:
            return obj

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"


# TODO: Look up path in some global [per-thread or thread-safe] list.
