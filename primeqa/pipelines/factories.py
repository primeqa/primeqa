import json
from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher


class SearcherFactory:
    _instances = {}

    @classmethod
    def get(cls, config) -> Searcher:
        instance_id = hash(json.dumps(config.__dict__, sort_keys=True))
        if instance_id not in cls._instances:
            cls._instances[instance_id] = Searcher(index="index", config=config)

        return cls._instances[instance_id]
