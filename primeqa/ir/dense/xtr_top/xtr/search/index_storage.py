import os
import sys
import pathlib

import torch
from math import ceil

from torch.utils.cpp_extension import load

from primeqa.ir.dense.xtr_top.xtr.indexing.loaders import load_doclens
from primeqa.ir.dense.xtr_top.xtr.indexing.codecs.residual_embeddings import ResidualEmbeddingsStrided

from primeqa.ir.dense.xtr_top.xtr.search.index_loader import IndexLoader
from primeqa.ir.dense.xtr_top.xtr.search.strided_tensor import StridedTensor
from primeqa.ir.dense.xtr_top.xtr.search.candidate_generation import CandidateGeneration


def score_reduce(scores_padded, D_mask):
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values.sum(-1)
    return scores

class IndexScorer(IndexLoader, CandidateGeneration):
    def __init__(self, index_path, config=None, use_gpu=True):
        super().__init__(index_path, config=config, use_gpu=use_gpu)

        IndexScorer.try_load_torch_extensions(use_gpu)

        self.embeddings_strided = ResidualEmbeddingsStrided(self.codec, self.embeddings, self.doclens)

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return

        verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True"

        print(f"Loading decompress_residuals_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...")
        try:
            decompress_residuals_cpp = load(
                name="decompress_residuals_cpp",
                sources=[
                    os.path.join(
                        pathlib.Path(__file__).parent.resolve(), "decompress_residuals.cpp"
                    ),
                ],
                extra_cflags=["-O3"],
                verbose=verbose,
            )
        except (RuntimeError, KeyboardInterrupt) as e:
            if not verbose:
                import traceback
                traceback.print_exc()
            #print_torch_extension_error_message()
            sys.exit(1)
        cls.decompress_residuals = decompress_residuals_cpp.decompress_residuals_cpp

        cls.loaded_extensions = True

    def lookup_eids(self, embedding_ids, codes=None, out_device='cuda'):
        return self.embeddings_strided.lookup_eids(embedding_ids.cpu(), codes=codes, out_device=out_device)

    def lookup_pids(self, passage_ids, out_device='cuda', return_mask=False):
        return self.embeddings_strided.lookup_pids(passage_ids, out_device)

    def embedding_ids_to_pids(self, embedding_ids):
        all_pids = torch.unique(self.emb2pid[embedding_ids.long()].cuda(), sorted=False)
        return all_pids

    def retrieve(self, config, Q, mask=None, k=10):
        Q = Q[:, :config.query_maxlen]   # NOTE: Candidate generation uses only the query tokens
        mask = mask[:, :config.query_maxlen]
        outputs = self.generate_candidates(config, Q, mask=mask, k=k)
        return outputs
