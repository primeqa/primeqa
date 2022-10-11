import primeqa.ir.dense.colbert_top.colbert.indexing.codecs.residual_embeddings as residual_embeddings

from primeqa.ir.dense.colbert_top.colbert.search.strided_tensor import StridedTensor

class ResidualEmbeddingsStrided:
    def __init__(self, codec, embeddings, doclens):
        self.codec = codec
        self.codes = embeddings.codes
        self.residuals = embeddings.residuals
        self.use_gpu = self.codec.use_gpu

        self.codes_strided = StridedTensor(self.codes, doclens, use_gpu=self.use_gpu)
        self.residuals_strided = StridedTensor(self.residuals, doclens, use_gpu=self.use_gpu)

    def lookup_eids(self, embedding_ids, codes=None, out_device='cuda'):
        codes = self.codes[embedding_ids] if codes is None else codes
        residuals = self.residuals[embedding_ids]

        return self.codec.decompress(residual_embeddings.ResidualEmbeddings(codes, residuals))

    def lookup_pids(self, passage_ids, out_device='cuda'):
        codes_packed, codes_lengths = self.codes_strided.lookup(passage_ids)
        residuals_packed, _ = self.residuals_strided.lookup(passage_ids)

        embeddings_packed = self.codec.decompress(residual_embeddings.ResidualEmbeddings(codes_packed, residuals_packed))

        return embeddings_packed, codes_lengths

    def lookup_codes(self, passage_ids):
        return self.codes_strided.lookup(passage_ids)
