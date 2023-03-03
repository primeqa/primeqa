from colbert.infra.run import Run
import os
import ujson

from colbert.evaluation.loaders import load_queries

class Queries:
    def __init__(self, path=None, data=None):
        self.path = path

        if data:
            assert isinstance(data, dict), type(data)
        self._load_data(data) or self._load_file(path)
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data.items())

    def provenance(self):
        return self.path
    
    def toDict(self):
        return {'provenance': self.provenance()}

    def _load_data(self, data):
        if data is None:
            return None

        self.data = {}
        self._qas = {}

        for qid, content in data.items():
            if isinstance(content, dict):
                self.data[qid] = content['question']
                self._qas[qid] = content
            else:
                self.data[qid] = content

        if len(self._qas) == 0:
            del self._qas

        return True

    def _load_file(self, path):
        if not path.endswith('.json'):
            self.data = load_queries(path)
            return True
        
        # Load QAs
        self.data = {}
        self._qas = {}

        with open(path) as f:
            for line in f:
                qa = ujson.loads(line)

                assert qa['qid'] not in self.data
                self.data[qa['qid']] = qa['question']
                self._qas[qa['qid']] = qa

        return self.data

    def qas(self):
        return dict(self._qas)

    def __getitem__(self, key):
        return self.data[key]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def save(self, new_path):
        assert new_path.endswith('.tsv')
        assert not os.path.exists(new_path), new_path

        with Run().open(new_path, 'w') as f:
            for qid, content in self.data.items():
                content = f'{qid}\t{content}\n'
                f.write(content)
            
            return f.name

    def save_qas(self, new_path):
        assert new_path.endswith('.json')
        assert not os.path.exists(new_path), new_path

        with open(new_path, 'w') as f:
            for qid, qa in self._qas.items():
                qa['qid'] = qid
                f.write(ujson.dumps(qa) + '\n')

    def _load_tsv(self, path):
        raise NotImplementedError

    def _load_jsonl(self, path):
        raise NotImplementedError

    @classmethod
    def cast(cls, obj):
        if type(obj) is str:
            return cls(path=obj)

        if isinstance(obj, dict) or isinstance(obj, list):
            return cls(data=obj)

        if type(obj) is cls:
            return obj

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"
