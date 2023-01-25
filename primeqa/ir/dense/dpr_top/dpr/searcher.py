import re
import torch
import csv
from typing import Union
import os
import numpy as np
import ujson as json
import logging

from transformers import (DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRQuestionEncoderTokenizerFast)

from primeqa.ir.dense.dpr_top.util.line_corpus import read_lines, write_open
from primeqa.ir.dense.dpr_top.util.reporting import Reporting
from primeqa.ir.dense.dpr_top.dpr.dpr_util import DPROptions, queries_to_vectors
from primeqa.ir.dense.dpr_top.util.args_help import fill_from_config
from primeqa.ir.dense.dpr_top.dpr.simple_mmap_dataset import Corpus
from primeqa.ir.dense.dpr_top.dpr.faiss_index import ANNIndex
from primeqa.ir.dense.dpr_top.dpr.config import DPRSearchArguments

logger = logging.getLogger(__name__)


class Options(DPROptions):
    def __init__(self):
        # from dpr_apply.__init__
        super().__init__()
        self.output_dir = ''
        self.kilt_data = ''
        self.top_k = 20  # we'll supply this many document ids for reporting provenance
        self.bsize = 32
        self.do_not_include_passages = False
        # ^ from dpr_apply

        # from corpus_server_direct.__init__
        self.index_location = ''
        # ^ from corpus_server_direct.__init__

        self.queries = ''
        self.query_file_type = 'id_text'
        self.__required_args__ = ['index_location', 'output_dir']
        self.output_json = False

class DPRSearcher():
    def __init__(self, config: DPRSearchArguments):
        # from dpr_apply.main
        self.opts = Options()
        fill_from_config(self.opts, config)
        torch.set_grad_enabled(False)
        self.report = Reporting()
        # ^ from dpr_apply.main

        # as in index_simple_corpus.py
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.opts.model_name_or_path != "":
            self.opts.qry_encoder_name_or_path = self.opts.model_name_or_path

        self.opts.qry_encoder_name_or_path = re.sub('\/config\.json$', '', self.opts.qry_encoder_name_or_path)

        self.qencoder = DPRQuestionEncoder.from_pretrained(self.opts.qry_encoder_name_or_path)
        self.qencoder = self.qencoder.to(self.device)
        self.qencoder.eval()
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.opts.qry_encoder_name_or_path)

        # from corpus_server_direct.run
        # we either have a single index.faiss or we have an index for each offsets/passages
        if os.path.exists(os.path.join(self.opts.index_location, "index.faiss")):
            self.passages = Corpus(os.path.join(self.opts.index_location))
            index = ANNIndex(os.path.join(self.opts.index_location, "index.faiss"))
            self.shards = None
            self.dim = index.dim()
        else:
            self.shards = []
            # loop over the different index*.faiss
            # so we have a list of (index, passages)
            # we search each index, then take the top-k results overall
            logger.info(f'Using sharded faiss, reading shards from {self.opts.index_location}')
            for filename in os.listdir(self.opts.index_location):
                if filename.startswith('passages') and filename.endswith('.json.gz.records'):
                    name = filename[len("passages"):-len(".json.gz.records")]
                    logger.info(f'Reading {filename}')
                    self.shards.append((ANNIndex(os.path.join(self.opts.index_location, f'index{name}.faiss')),
                                   Corpus(os.path.join(self.opts.index_location, f'passages{name}.json.gz.records'))))
            self.dim = self.shards[0][0].dim()
            assert all([self.dim == shard[0].dim() for shard in self.shards])
            logger.info(f'Using sharded faiss with {len(self.shards)} shards.')
        self.dummy_doc = {'pid': 'N/A', 'title': '', 'text': '', 'vector': np.zeros(self.dim, dtype=np.float32)}


    def search(self, query_batch = None, top_k = 10, mode: Union['query_list', 'queries_and_results_in_files', None] = None):
        # from corpus_server_direct.run
        def _get_docs_by_pids(pids, *, dummy_if_missing=False):
            docs = []
            for pid in pids:
                doc = None
                if self.shards is None:
                    doc = self.passages.get_by_pid(pid)
                else:
                    for shard in self.shards:
                        doc = shard[1].get_by_pid(pid)
                        if doc is not None:
                            break
                if doc is None:
                    if dummy_if_missing:
                        doc = self.dummy_doc
                    else:
                        raise ValueError
                docs.append(doc)
            return docs

        def merge_results(query_vectors, k): # from corpus_server_direct.merge_results
            # CONSIDER: consider ResultHeap (https://github.com/matsui528/faiss_tips)
            all_scores = np.zeros((query_vectors.shape[0], k * len(self.shards)), dtype=np.float32)
            all_indices = np.zeros((query_vectors.shape[0], k * len(self.shards)), dtype=np.int64)
            for si, shard in enumerate(self.shards):
                index_i, passages_i = shard
                scores, indexes = index_i.search(query_vectors, k)
                assert len(scores.shape) == 2
                assert scores.shape[1] == k
                assert scores.shape == indexes.shape
                assert scores.dtype == np.float32
                assert indexes.dtype == np.int64
                all_scores[:, si * k: (si + 1) * k] = scores
                all_indices[:, si * k: (si + 1) * k] = indexes
            kbest = all_scores.argsort()[:, -k:][:, ::-1]
            docs = [[self.shards[ndx // k][1][all_indices[bi, ndx]] for ndx in ndxs] for bi, ndxs in enumerate(kbest)]
            return docs

        # from dpr_apply
        def retrieve(queries):
            with torch.no_grad():
                query_vectors_tensor = queries_to_vectors(self.tokenizer, self.qencoder, queries)

                # from from corpus_server_direct.retrieve_docs
                query_vectors = query_vectors_tensor.detach().cpu().numpy().astype(np.float32)
                batch_size = query_vectors.shape[0]
                assert query_vectors.shape[1] == self.dim

                if self.shards is None:
                    scores, indexes = self.index.search(query_vectors, self.opts.top_k)
                    docs = [[self.passages[ndx] for ndx in ndxs] for ndxs in indexes]
                else:
                    docs = merge_results(query_vectors, self.opts.top_k)

                doc_dicts = [{'pid': [dqk['pid'] for dqk in dq],
                              'title': [dqk['title'] for dqk in dq],
                              'text': [dqk['text'] for dqk in dq]} for dq in docs]

                doc_vectors = np.zeros([batch_size, self.opts.top_k, self.dim], dtype=np.float32)
                for qi, docs_qi in enumerate(docs):
                    gpids = []
                    for ki, doc_qi_ki in enumerate(docs_qi):
                        # if we have gold_pids, set their vector to 100 * the query vector
                        if ki < len(gpids):
                            doc_vectors[qi, ki] = 100 * query_vectors[qi]
                        else:
                            doc_vectors[qi, ki] = doc_qi_ki['vector']
                # ^ from from corpus_server_direct.retrieve_docs

                # from corpus_client.retrieve
                retrieved_doc_embeds = torch.Tensor(doc_vectors.copy()).to(query_vectors_tensor)
                doc_scores = torch.bmm(
                    query_vectors_tensor.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                ).squeeze(1)
                # ^ from corpus_client.retrieve

            # from dpr_apply.retrieve
            # after
            # doc_scores, docs, doc_vectors = client_retrieve(query_vectors, n_docs=opts.n_docs_for_provenance)
            doc_scores = doc_scores.detach().cpu().numpy()
            docs = doc_dicts # Because in corpus_server_directretrieve_docs: "retval = {'docs': doc_dicts}"

            retrieved_doc_ids = [dd['pid'] for dd in docs]

            passages = None
            if not self.opts.do_not_include_passages:
                passages = [{'titles': dd['title'], 'texts': dd['text'], 'scores': doc_scores[dndx].tolist()} for dndx, dd in enumerate(docs)]

            return retrieved_doc_ids, passages
            # ^ from dpr_apply.retrieve

        # from convert_for_kilt_eval
        def to_distinct_doc_ids(passage_ids):
            doc_ids = []
            for pid in passage_ids:
                doc_id = pid[:pid.find(':')]
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)
            return doc_ids

        # from dpr_apply
        def record_one_instance(output, inst_id, input, doc_ids, passages):
            if self.opts.output_json:
                wids = to_distinct_doc_ids(doc_ids)
                pred_record = {'id': inst_id, 'input': input, 'output': [{'answer': '', 'provenance': [{'wikipedia_id': wid} for wid in wids]}]}
                if passages:
                    pred_record['passages'] = [{'pid': pid, 'title': title, 'text': text, 'score': float(score)}
                                               for pid, title, text, score in zip(doc_ids, passages['titles'], passages['texts'], passages['scores'])]
                output.write(json.dumps(pred_record, indent=4) + '\n')
            else:
                 for rank, doc_id in enumerate(doc_ids):
                    output.writerow([inst_id, doc_id, rank, passages['scores'][rank]])

            if self.report.is_time():
               logger.info(f'Finished instance {self.report.check_count}, {self.report.check_count/self.report.elapsed_seconds()} per second.')

        # from dpr_apply
        def one_batch(id_batch, query_batch, output):
            retrieved_doc_ids, passages = retrieve(query_batch)
            for bi in range(len(query_batch)):
                record_one_instance(output, id_batch[bi], query_batch[bi], retrieved_doc_ids[bi], passages[bi] if passages else None)

        # from dpr_apply.main
        if self.opts.world_size > 1:
            raise NotImplementedError(f'Distributed not supported (yet).')
        if mode == None or mode == 'queries_and_results_in_files':
            if not os.path.exists(self.opts.output_dir):
                logger.info(f'Creating directory {self.opts.output_dir}')
                os.makedirs(self.opts.output_dir)

            with write_open(os.path.join(self.opts.output_dir,  'ranked_passages.tsv')) as output_fh:
                if self.opts.output_json:
                    output = output_fh
                else:
                    output = csv.writer(output_fh, delimiter="\t", quotechar='"')
                id_batch, query_batch = [], []
                for line_ndx, line in enumerate(read_lines(self.opts.queries)):
                    if self.opts.query_file_type == 'id_text':
                        [qry_id, qry_text] = next(csv.reader([line], delimiter="\t", quotechar='"'))
                    elif self.opts.query_file_type == 'text_answers':
                        [qry_text, answers] = next(csv.reader([line], delimiter="\t", quotechar='"'))
                        qry_id = line_ndx
                    else:
                        raise NotImplementedError(f"Query file type {self.opts.query_file_type} is not implemented (yet).")
                    id_batch.append(qry_id)
                    query_batch.append(qry_text)
                    if len(query_batch) == self.opts.bsize:
                        one_batch(id_batch, query_batch, output)
                        id_batch, query_batch = [], []
                if len(query_batch) > 0:
                    one_batch(id_batch, query_batch, output)
            logger.info(f'Finished instance {self.report.check_count}, {self.report.check_count/self.report.elapsed_seconds()} per second.')
        elif mode == 'query_list':
            self.opts.top_k = top_k
            retrieved_doc_ids, passages = retrieve(query_batch)
            # retrieved_doc_ids: topN list of lists of doc IDs as strings
            # passages: topN list of dicts {'titles', 'texts', 'scores' as floats}
            return retrieved_doc_ids, passages
        else:
            raise NotImplementedError(f'Mode {mode} is not supported (yet).')
