from primeqa.ir.dense.dpr_top.util.line_corpus import write_open
import ujson as json
import logging
import os
import base64
import torch
from typing import List
import numpy as np
import re

from primeqa.ir.dense.dpr_top.dpr.simple_mmap_dataset import gzip_str
from primeqa.ir.dense.dpr_top.dpr.faiss_index import build_index, IndexOptions

from primeqa.ir.dense.dpr_top.util.reporting import Reporting
from primeqa.ir.util.corpus_reader import corpus_reader, Passage
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
)

from primeqa.ir.dense.dpr_top.util.args_help import fill_from_config
from primeqa.ir.dense.dpr_top.dpr.config import DPRIndexingArguments

logger = logging.getLogger(__name__)

class Options(IndexOptions):
    def __init__(self):
        super().__init__()
        self.ctx_encoder_name_or_path = 'facebook/dpr-ctx_encoder-multiset-base'
        self.embed = '1of1'
        self.sharded_index = True
        self.collection = ''
        self.output_dir = ''  # the output_dir will have the passages dataset and the hnsw_index.faiss
        self.bsize = 16
        self.__required_args__ = ['output_dir']
        self.max_doc_length=128 # to match dataloader_biencoder.make_batch : self.ctx_tokenizer(ctx_titles, ctx_texts

class DPRIndexer():
    def __init__(self, config: DPRIndexingArguments):
        self.opts = Options()
        fill_from_config(self.opts, config)

        torch.set_grad_enabled(False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.opts.output_dir, exist_ok=True)

        self.embed_num, self.embed_count = [int(n.strip()) for n in self.opts.embed.split('of')]
        assert 1 <= self.embed_num <= self.embed_count

        self.opts.ctx_encoder_name_or_path = re.sub('\/config\.json$', '', self.opts.ctx_encoder_name_or_path)
        self.ctx_encoder = DPRContextEncoder.from_pretrained(self.opts.ctx_encoder_name_or_path).to(device=self.device)
        self.ctx_encoder.eval()
        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(self.opts.ctx_encoder_name_or_path)


    def embed(self, doc_batch: List[Passage], ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> np.ndarray:
        documents = {"title": [doci.title if doci.title is not None else "" for doci in doc_batch], 'text': [doci.text for doci in doc_batch]}
        """Compute the DPR embeddings of document passages"""
        input_ids = ctx_tokenizer(
            documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt", max_length=self.opts.max_doc_length
        )["input_ids"]
        embeddings = ctx_encoder(input_ids.to(device=self.device), return_dict=True).pooler_output
        return embeddings.detach().cpu().to(dtype=torch.float16).numpy()


    def write(self, cur_offset, offsets, passage_file, doc_batch: List[Passage], embeddings):
        assert len(doc_batch) == embeddings.shape[0]
        assert len(embeddings.shape) == 2
        for di, doc in enumerate(doc_batch):
            doc = doc.to_dict()
            doc['vector'] = base64.b64encode(embeddings[di].astype(np.float16)).decode('ascii')
            jstr_gz = gzip_str(json.dumps(doc))
            offsets.append(cur_offset)
            passage_file.write(jstr_gz)
            cur_offset += len(jstr_gz)
        return cur_offset


    def index(self):
        offsets = []
        cur_offset = 0
        passages = write_open(os.path.join(self.opts.output_dir, f'passages_{self.embed_num}_of_{self.embed_count}.json.gz.records'), binary=True)

        report = Reporting()
        doc_batch = []
        for pndx, passage in enumerate(corpus_reader(self.opts.collection, fieldnames = ('id', 'text', 'title'))):
            if pndx == 0 and (passage.pid == 'id' or passage.pid == 'pid') and (passage.text == 'text' or passage.text == 'contents') and passage.title == 'title':
                continue
            if pndx % self.embed_count != (self.embed_num-1):
                continue
            if report.is_time():
                logger.info(f'on instance {report.check_count}, {report.check_count/report.elapsed_seconds()} instances per second')
            doc_batch.append(passage)
            if len(doc_batch) == self.opts.bsize:
                embeddings = self.embed(doc_batch, self.ctx_encoder, self.ctx_tokenizer)
                cur_offset = self.write(cur_offset, offsets, passages, doc_batch, embeddings)
                doc_batch = []
        if len(doc_batch) > 0:
            embeddings = self.embed(doc_batch, self.ctx_encoder, self.ctx_tokenizer)
            cur_offset = self.write(cur_offset, offsets, passages, doc_batch, embeddings)
        offsets.append(cur_offset)  # just the length of the file
        passages.close()
        with write_open(os.path.join(self.opts.output_dir, f'offsets_{self.embed_num}_of_{self.embed_count}.npy'), binary=True) as f:
            np.save(f, np.array(offsets, dtype=np.int64), allow_pickle=False)
        logger.info(f'wrote passages_{self.embed_num}_of_{self.embed_count}.json.gz.records in {report.elapsed_time_str()}')
        #print(f'Wrote passages_{self.embed_num}_of_{self.embed_count}.json.gz.records in {report.elapsed_time_str()}')

        if self.opts.sharded_index:
            build_index(os.path.join(self.opts.output_dir, f'passages_{self.embed_num}_of_{self.embed_count}.json.gz.records'),
                        os.path.join(self.opts.output_dir, f'index_{self.embed_num}_of_{self.embed_count}.faiss'), self.opts)
        elif self.embed_count == 1:
            build_index(self.opts.output_dir, os.path.join(self.opts.output_dir, 'index.faiss'), self.opts)
