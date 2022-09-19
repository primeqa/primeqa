from transformers import (DPRQuestionEncoder, DPRQuestionEncoderTokenizer)
import torch
import csv
from typing import List
import os
import numpy as np

from primeqa.ir.dense.dpr_top.util.line_corpus import read_lines, write_open
import ujson as json
from primeqa.ir.dense.dpr_top.util.reporting import Reporting
import logging
from primeqa.ir.dense.dpr_top.dpr.dpr_util import DPROptions, queries_to_vectors
from primeqa.ir.dense.dpr_top.util.args_help import fill_from_args
from primeqa.ir.dense.dpr_top.dpr.simple_mmap_dataset import Corpus
from primeqa.ir.dense.dpr_top.dpr.faiss_index import ANNIndex

logger = logging.getLogger(__name__)


class Options(DPROptions):
    def __init__(self):
        # from dpr_apply.__init__
        super().__init__()
        self.output = ''
        self.kilt_data = ''
        self.n_docs_for_provenance = 20  # we'll supply this many document ids for reporting provenance
        self.retrieve_batch_size = 32
        self.include_passages = False  # if set, we return the list of passages too
        # ^ from dpr_apply

        # from corpus_server_direct.__init__
        self.corpus_dir = ''
        # ^ from corpus_server_direct.__init__

        self.qry_tokenizer_path = ''
        self.queries = ''

        self.query_file_type = 'id_text'

        self.__required_args__ = ['corpus_dir', 'output']

        # for compatibility with run_ir.py
        self.engine_type = 'DPR'
        self.do_search = False

        self.output_simple_tsv = False

class DPRSearcher():
    def __init__(self):
        # from dpr_apply.main
        self.opts = Options()
        fill_from_args(self.opts)
        torch.set_grad_enabled(False)
        self.report = Reporting()
        # ^ from dpr_apply.main

        # as in index_simple_corpus.py
        # TODO: compare with in BiEncoderTrainer: "self.args = BiEncoderTrainArgs().fill_from_args()"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.qencoder = DPRQuestionEncoder.from_pretrained(self.opts.qry_encoder_path)
        self.qencoder.eval()
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.opts.qry_tokenizer_path)

        # from corpus_server_direct.run
        # we either have a single index.faiss or we have an index for each offsets/passages
        if os.path.exists(os.path.join(self.opts.corpus_dir, "index.faiss")):
            self.passages = Corpus(os.path.join(self.opts.corpus_dir))
            index = ANNIndex(os.path.join(self.opts.corpus_dir, "index.faiss"))
            self.shards = None
            self.dim = index.dim()
        else:
            self.shards = []
            # loop over the different index*.faiss
            # so we have a list of (index, passages)
            # we search each index, then take the top-k results overall
            logger.info(f'Using sharded faiss, reading shards from {self.opts.corpus_dir}')
            for filename in os.listdir(self.opts.corpus_dir):
                if filename.startswith('passages') and filename.endswith('.json.gz.records'):
                    name = filename[len("passages"):-len(".json.gz.records")]
                    logger.info(f'Reading {filename}')
                    self.shards.append((ANNIndex(os.path.join(self.opts.corpus_dir, f'index{name}.faiss')),
                                   Corpus(os.path.join(self.opts.corpus_dir, f'passages{name}.json.gz.records'))))
            self.dim = self.shards[0][0].dim()
            assert all([self.dim == shard[0].dim() for shard in self.shards])
            logger.info(f'Using sharded faiss with {len(self.shards)} shards.')
        self.dummy_doc = {'pid': 'N/A', 'title': '', 'text': '', 'vector': np.zeros(self.dim, dtype=np.float32)}

    def search(self):

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
                    scores, indexes = self.index.search(query_vectors, self.opts.n_docs_for_provenance)
                    docs = [[self.passages[ndx] for ndx in ndxs] for ndxs in indexes]
                else:
                    docs = merge_results(query_vectors, self.opts.n_docs_for_provenance)

                if 'pid' in docs[0][0]:
                    doc_dicts = [{'pid': [dqk['pid'] for dqk in dq],
                                  'title': [dqk['title'] for dqk in dq],
                                  'text': [dqk['text'] for dqk in dq]} for dq in docs]
                else:
                    doc_dicts = [{'title': [dqk['title'] for dqk in dq],
                                  'text': [dqk['text'] for dqk in dq]} for dq in docs]

                include_vectors = True # TODO: make this an arg? Note: this is "if not only_docs" in corpus_client.retrieve
                if include_vectors:
                    doc_vectors = np.zeros([batch_size, self.opts.n_docs_for_provenance, self.dim], dtype=np.float32)
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
                if include_vectors: # this is "if not only_docs" in corpus_client.retrieve
                    #doc_vectors = np.frombuffer(base64.decodebytes(rdocs['doc_vectors'].encode('ascii')), dtype=self.rest_dtype).\
                    #    reshape(-1, n_docs_for_provenance, question_encoder_last_hidden_state.shape[-1])[:, 0:n_docs, :]
                    retrieved_doc_embeds = torch.Tensor(doc_vectors.copy()).to(query_vectors_tensor)
                    doc_scores = torch.bmm(
                        query_vectors_tensor.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                    ).squeeze(1)
                else:
                    doc_scores = None
                    retrieved_doc_embeds = None
                # ^ from corpus_client.retrieve

            # from dpr_apply.retrieve
            # after
            # doc_scores, docs, doc_vectors = client_retrieve(query_vectors, n_docs=opts.n_docs_for_provenance)
            doc_scores = doc_scores.detach().cpu().numpy()
            docs = doc_dicts # Because in corpus_server_directretrieve_docs: "retval = {'docs': doc_dicts}"

            if 'id' in docs[0]:
                retrieved_doc_ids = [dd['id'] for dd in docs]
            elif 'pid' in docs[0]:
                retrieved_doc_ids = [dd['pid'] for dd in docs]
            else:
                retrieved_doc_ids = [[0] * len(dd['text']) for dd in docs]  # dummy ids
            passages = None
            if self.opts.include_passages:
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
            if self.opts.output_simple_tsv:
                for rank, doc_id in enumerate(doc_ids):
                    output.writerow([inst_id, doc_id, rank, passages['scores'][rank]])
            else:
                wids = to_distinct_doc_ids(doc_ids)
                pred_record = {'id': inst_id, 'input': input, 'output': [{'answer': '', 'provenance': [{'wikipedia_id': wid} for wid in wids]}]}
                if passages:
                    pred_record['passages'] = [{'pid': pid, 'title': title, 'text': text, 'score': float(score)}
                                               for pid, title, text, score in zip(doc_ids, passages['titles'], passages['texts'], passages['scores'])]
                output.write(json.dumps(pred_record, indent=4) + '\n')
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
        with write_open(self.opts.output) as output_fh:
            if self.opts.output_simple_tsv:
                output = csv.writer(output_fh, delimiter="\t", quotechar='"')
            else:
                output = output_fh
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
                if len(query_batch) == self.opts.retrieve_batch_size:
                    one_batch(id_batch, query_batch, output)
                    id_batch, query_batch = [], []
            if len(query_batch) > 0:
                one_batch(id_batch, query_batch, output)
        logger.info(f'Finished instance {self.report.check_count}, {self.report.check_count/self.report.elapsed_seconds()} per second.')
