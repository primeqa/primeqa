import os
import json
import ujson
import numpy as np

import torch

from primeqa.ir.dense.xtr_top.xtr.indexing.utils import optimize_ivf
from primeqa.ir.dense.xtr_top.xtr.search.strided_tensor import StridedTensor
from primeqa.ir.dense.xtr_top.xtr.indexing.codecs.residual import ResidualCodec


class IndexLoader:
    def __init__(self, index_path, config=None, use_gpu=torch.cuda.is_available()):
        self.index_path = index_path
        self.use_gpu = use_gpu

        self._load_codec(config)
        self._load_ivf()

        self._load_doclens()
        self._load_embeddings()
        self._load_empty_cluster_ids(use_gpu)
        self._load_mappings()

    def _load_codec(self, config):
        print(f"#> Loading codec...")
        self.codec = ResidualCodec.load(self.index_path, config=config)

    def _load_mappings(self):
        print(f"#> Loading mappings...")
        with open(f"{self.index_path}/mappings.json") as fp:
            self.mappings = json.load(fp)
            indices = []
            flag = set([])
            for index, docstr in enumerate(self.mappings):
                if docstr in flag:
                    indices.append(indices[-1])
                else:
                    flag.add(docstr)
                    indices.append(index)
                    
        if self.use_gpu:
            self.mappings = {"doc_strings": np.array(self.mappings), "doc_indices": np.array(indices)}
        else:
            self.mappings = {"doc_strings": np.array(self.mappings), "doc_indices": np.array(indices)}

    def _load_ivf(self):
        print(f"#> Loading IVF...")

        ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.eid.pt"), map_location='cpu')

        if os.path.exists(os.path.join(self.index_path, "emb2pid.pt")):
            self.emb2pid = torch.load(os.path.join(self.index_path, "emb2pid.pt"), map_location='cpu')
            if self.use_gpu:
                self.emb2pid = self.emb2pid.cuda()

        #ivf = StridedTensor(ivf, ivf_lengths, use_gpu=self.use_gpu)

        if self.use_gpu:
            self.ivf = ivf.cuda() #.split(ivf_lengths.tolist())
            self.ivf_lengths = torch.cumsum(ivf_lengths, 0).cuda()
            self.ivf_lengths_prev = torch.cat([torch.tensor([0]).cuda(), self.ivf_lengths[:-1]])
            self.ivf_ranges = [torch.arange(pcell, cell) for pcell, cell in zip(self.ivf_lengths_prev, self.ivf_lengths)]
        else:
            self.ivf = ivf
            self.ivf_lengths = torch.cumsum(ivf_lengths, 0)
            self.ivf_lengths_prev = torch.cat([torch.tensor([0]), self.ivf_lengths])
            self.ivf_ranges = [torch.arange(pcell, cell) for pcell, cell in zip(self.ivf_lengths_prev, self.ivf_lengths)]

    def _load_empty_cluster_ids(self, use_gpu=False):
        with open(os.path.join(self.index_path, "empty_clusters.json")) as fp:
            self.empty_clusters = torch.tensor(json.load(fp))
        if use_gpu:
            self.empty_clusters = self.empty_clusters.cuda()

    def _load_doclens(self):
        doclens = []

        for chunk_idx in range(self.num_chunks):
            with open(os.path.join(self.index_path, f'doclens.{chunk_idx}.json')) as f:
                chunk_doclens = ujson.load(f)
                doclens.extend(chunk_doclens)

        self.doclens = torch.tensor(doclens)

    def _load_embeddings(self):
        self.embeddings = ResidualCodec.Embeddings.load_chunks(self.index_path, range(self.num_chunks),
                                                               self.num_embeddings)

    @property
    def metadata(self):
        try:
            self._metadata
        except:
            with open(os.path.join(self.index_path, 'metadata.json')) as f:
                self._metadata = ujson.load(f)

        return self._metadata

    @property
    def config(self):
        raise NotImplementedError()  # load from dict at metadata['config']

    @property
    def num_chunks(self):
        # EVENTUALLY: If num_chunks doesn't exist (i.e., old index), fall back to counting doclens.*.json files.
        return self.metadata['num_chunks']

    @property
    def num_embeddings(self):
        # EVENTUALLY: If num_embeddings doesn't exist (i.e., old index), sum the values in doclens.*.json files.
        return self.metadata['num_embeddings']
