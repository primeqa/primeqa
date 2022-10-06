import os
import ujson
import torch
import numpy as np

from primeqa.ir.dense.colbert_top.colbert.utils.utils import lengths2offsets, print_message, dotdict, flatten
from primeqa.ir.dense.colbert_top.colbert.indexing.codecs.residual import ResidualCodec
from primeqa.ir.dense.colbert_top.colbert.indexing.utils import optimize_ivf
from primeqa.ir.dense.colbert_top.colbert.search.strided_tensor import StridedTensor


class IndexLoader:
    def __init__(self, index_path, use_gpu=torch.cuda.is_available()):
        self.index_path = index_path
        self.use_gpu = use_gpu

        self._load_codec()
        self._load_ivf()

        self._load_doclens()
        self._load_embeddings()

    def _load_codec(self):
        print_message(f"#> Loading codec...")
        self.codec = ResidualCodec.load(self.index_path)

    def _load_ivf(self):
        print_message(f"#> Loading IVF...")

        if os.path.exists(os.path.join(self.index_path, "ivf.pid.pt")):
            ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pid.pt"), map_location='cpu')
        else:
            assert os.path.exists(os.path.join(self.index_path, "ivf.pt")), f"ivf.pt not found in {self.index_path}"
            ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pt"), map_location='cpu')
            ivf, ivf_lengths = optimize_ivf(ivf, ivf_lengths, self.index_path)

        ivf = StridedTensor(ivf, ivf_lengths, use_gpu=self.use_gpu)

        self.ivf = ivf

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

