import torch

from primeqa.ir.dense.xtr_top.xtr.indexing.utils import batch

class CollectionEncoder():
    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint

    def encode_passages(self, passages):
        print(f"#> Encoding {len(passages)} passages..")

        if len(passages) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []

            # Batch here to avoid OOM from storing intermediate embeddings on GPU.
            # Storing on the GPU helps with speed of masking, etc.
            # But ideally this batching happens internally inside docFromText.
            for passages_batch in batch(passages, self.config.bsize * 50):
            # for passages_batch in batch(passages, self.config.bsize * 1):
                #doclens_ -> number of tokens in each document in a batch
                embs_, doclens_ = self.checkpoint.docFromText(passages_batch, bsize=self.config.bsize,
                                                              keep_dims='flatten', showprogress=False)
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)

        return embs, doclens
