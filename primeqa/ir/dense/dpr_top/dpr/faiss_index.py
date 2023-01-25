import numpy as np
from primeqa.ir.dense.dpr_top.dpr.simple_mmap_dataset import Corpus
from primeqa.ir.dense.dpr_top.util.args_help import fill_from_args
import faiss
from primeqa.ir.dense.dpr_top.util.reporting import Reporting
import os
import time
import math
import logging

logger = logging.getLogger(__name__)

def l2_convert_indexed_vectors(vectors: np.ndarray, max_norm_sqrd: float):
    norms = np.linalg.norm(vectors, axis=1)
    aux_dims_sqrd = max_norm_sqrd - norms * norms
    if np.min(aux_dims_sqrd) < 0:
        print(f'WARNING: max_norm_sqrd = {max_norm_sqrd} but it was less '
              f'({np.min(aux_dims_sqrd)}) than a vectors norm_sqrd')
        aux_dims_sqrd = np.maximum(aux_dims_sqrd, 0)
    aux_dims = np.sqrt(aux_dims_sqrd)
    converted_vectors = np.hstack((vectors, aux_dims.reshape(-1, 1)))
    return converted_vectors


def l2_convert_query_vectors(vectors: np.ndarray):
    aux_dim = np.zeros(vectors.shape[0], dtype=np.float32)
    converted_vectors = np.hstack((vectors, aux_dim.reshape(-1, 1)))
    return converted_vectors


class ANNIndex:
    def __init__(self, index_file):
        self.index = faiss.read_index(index_file)
        self.is_l2 = type(self.index) == faiss.IndexHNSWSQ

    def search(self, query_vectors, k):
        if self.is_l2:
            query_vectors = l2_convert_query_vectors(query_vectors)
        scores, indexes = self.index.search(query_vectors, k)
        if self.is_l2:
            scores = -1 * scores  # make higher scores better always
        return scores, indexes

    def dim(self):
        if self.is_l2:
            return self.index.d - 1
        else:
            return self.index.d


class IndexOptions():
    def __init__(self):
        self.d = 768
        self.m = 128
        self.ef_search = 128
        self.ef_construction = 200
        self.index_batch_size = 100000
        self.scalar_quantizer = -1
        self.product_quantizer_m = -1  # suggested values of 8, 16, 32 - maybe we need much higher though
        self.product_quantizer_sv_bits = 8  # probably don't change this
        self.is_l2 = False
        self.num_vectors = -1
        self.max_norm = -1

    def _post_argparse(self):
        self.is_l2 = self.scalar_quantizer > 0


def build_index(corpus_dir, output_file, opts: IndexOptions):
    logger.info(f'building index, reading data from {corpus_dir}, writing to {output_file}')
    corpus = Corpus(corpus_dir)
    if opts.is_l2:
        print(f'Using L2 distance conversion')

    if (opts.num_vectors <= 0 and opts.product_quantizer_m > 0) or (opts.max_norm <= 0 and opts.is_l2):
        max_norm = 0
        num_vectors = 0
        start_time = time.time()
        for psg in corpus:
            vector = psg['vector']
            max_norm = max(max_norm, np.linalg.norm(vector))
            num_vectors += 1
        print(f'found max norm = {max_norm} over {num_vectors} vectors in {(time.time()-start_time)/60} min.')
        opts.max_norm = max_norm
        opts.num_vectors = num_vectors
    max_norm_sqrd = opts.max_norm * opts.max_norm

    opts.is_trained = False
    if opts.scalar_quantizer > 0:
        if opts.scalar_quantizer == 16:
            sq = faiss.ScalarQuantizer.QT_fp16
        elif opts.scalar_quantizer == 8:
            sq = faiss.ScalarQuantizer.QT_8bit
        elif opts.scalar_quantizer == 4:
            sq = faiss.ScalarQuantizer.QT_4bit
        elif opts.scalar_quantizer == 6:
            sq = faiss.ScalarQuantizer.QT_6bit
        else:
            raise ValueError(f'unknown --scalar_quantizer {opts.scalar_quantizer}')
        # see https://github.com/facebookresearch/faiss/blob/master/benchs/bench_hnsw.py
        index = faiss.IndexHNSWSQ(opts.d+1, sq, opts.m)
        index.hnsw.efSearch = opts.ef_search
        index.hnsw.efConstruction = opts.ef_construction
    elif opts.product_quantizer_m > 0:
        # product quant: https://github.com/matsui528/faiss_tips
        # seems awful - maybe product_quantizer_m should be much higher?
        nlist = int(math.sqrt(opts.num_vectors))
        quantizer = faiss.IndexHNSWFlat(opts.d, opts.m, faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexIVFPQ(quantizer, opts.d, nlist, opts.product_quantizer_m,
                                 opts.product_quantizer_sv_bits, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = opts.ef_search  # matsui528 recommended 8
        opts.index_batch_size = max(opts.index_batch_size, 50 * nlist)
    else:
        index = faiss.IndexHNSWFlat(opts.d, opts.m, faiss.METRIC_INNER_PRODUCT)
        # defaults are 16 and 40
        index.hnsw.efSearch = opts.ef_search
        index.hnsw.efConstruction = opts.ef_construction
        opts.is_trained = True  # doesn't need training

    vectors = np.zeros((opts.index_batch_size, opts.d), dtype=np.float32)
    vector_ndx = 0

    def add_to_index(vectors):
        if opts.is_l2:
            to_index = l2_convert_indexed_vectors(vectors, max_norm_sqrd)
        else:
            to_index = vectors
        if not opts.is_trained:
            index.train(to_index)
            opts.is_trained = True
        logger.info(f'calling index.add with {len(to_index)} vectors')
        index.add(to_index)

    report = Reporting()

    for pndx, psg in enumerate(corpus):
        if (pndx % 100000) == 0:
            logger.info(f'processed {pndx} passages')
        if report.is_time():
            print(report.progress_str(instance_name='vector'))
        vector = psg['vector']
        vectors[vector_ndx] = vector
        vector_ndx += 1
        if vector_ndx == opts.index_batch_size:
            logger.info(f'processed {pndx} passages')
            add_to_index(vectors)
            vector_ndx = 0
    if vector_ndx > 0:
        add_to_index(vectors[:vector_ndx])
    logger.info(f'processed {len(corpus)} passages')
    logger.info(f'finished building index, writing index file to {output_file}')
    #print(f'finished building index, writing index file to {output_file}')
    faiss.write_index(index, output_file)
    logger.info(f'took {report.elapsed_time_str()}')
    #print(f'took {report.elapsed_time_str()}')


if __name__ == "__main__":
    class CmdOptions(IndexOptions):
        def __init__(self):
            super().__init__()
            self.collection = ''  # can be a directory with passages*.json.gz.records or a single such file
            self.__required_args__ = ['collection']

    opts = CmdOptions()
    fill_from_args(opts)

    if os.path.isdir(opts.collection):
        output_file = os.path.join(opts.collection, 'index.faiss')
    else:
        base_dir, filename = os.path.split(opts.collection)
        assert filename.startswith('passages') and filename.endswith('.json.gz.records')
        index_fname = f'index{filename[len("passages"):-len(".json.gz.records")]}.faiss'
        output_file = os.path.join(base_dir, index_fname)

    build_index(opts.collection, output_file, opts)
