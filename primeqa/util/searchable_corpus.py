import os
import tempfile
from typing import List, AnyStr, Union
from unittest.mock import patch
import sys

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from primeqa.ir.dense.dpr_top.dpr.config import DPRIndexingArguments
from primeqa.ir.dense.dpr_top.dpr.index_simple_corpus import DPRIndexer
from primeqa.ir.dense.dpr_top.dpr.searcher import DPRSearcher
from primeqa.ir.dense.dpr_top.dpr.config import DPRSearchArguments
from primeqa.ir.dense.colbert_top.colbert.indexer import Indexer
from primeqa.ir.dense.colbert_top.colbert.infra import Run, RunConfig
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.utils.parser import Arguments
from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher
from primeqa.ir.dense.dpr_top.dpr.dpr_util import queries_to_vectors
from transformers import (
    HfArgumentParser,
    DPRContextEncoderTokenizer
)


class SearchableCorpus:
    def __init__(self, model_name, batch_size=64, top_k=10):
        self._is_dpr = True
        self.top_k = top_k

        if not os.path.exists(model_name): # Assume a HF model name
            model_name = hf_hub_download(repo_id=model_name, filename="config.json")
        self.model_name = model_name
        if os.path.exists(os.path.join(model_name,"ctx_encoder")): # Looks like a DPR model
            self._is_dpr = True
        else:
            self._is_colbert = True
        self.ctxt_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            os.path.join(self.model_name,"ctx_encoder"))
        # self.qry_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        #     os.path.join(self.model_name,"qry_encoder"))


    def add(self, texts:List[AnyStr], titles:List[AnyStr]=None, ids:List[AnyStr]=None, **kwargs):
        """
        Adds documents to the collection, including optionally the titles and the ids of the indexed items
        (possibly passages).
        Args:
            - texts:List[AnyStr] - a list of documents to be indexed
            - titles: List[AnyStr] - the list of titles for the texts to be indexed. These will be added to the text
                                     before indexing.
            - ids: List[AnyStr] - the list of ids for the texts. By default, they will be the position in the list.
        Returns:
            Nothing
        """
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.working_dir = self.tmp_dir.name
        self.output_dir = os.path.join(self.working_dir, 'output_dir')
        os.makedirs(os.path.join(self.working_dir, "input_dir"))
        self.input_passages = os.path.join(self.working_dir, "input_dir", "input.tsv")
        with open(self.input_passages, "w") as w:
            for i, t in enumerate(texts):
                w.write("\t".join([
                    str(i + 1) if ids is None else ids[i],
                    texts[i].strip(),
                    titles[i] if titles is not None else ""
                ]) + "\n"
                )
        if self._is_dpr:
            # embs = self.encode(texts, self.ctxt_tokenizer, kwargs['batch_size'] if 'batch_size' in kwargs else 64)

            index_args = [
                "prog",
                "--bsize", "16",
                "--ctx_encoder_name_or_path", os.path.join(self.model_name, "ctx_encoder"),
                "--embed", "1of1",
                "--output_dir", os.path.join(self.output_dir, "index"),
                "--collection", self.input_passages,
            ]

            parser = HfArgumentParser(DPRIndexingArguments)
            (dpr_args, remaining_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True, args=index_args)

            self.indexer = DPRIndexer(dpr_args)
            self.indexer.index()

            search_args = [
                "prog",
                "--engine_type", "DPR",
                "--model_name_or_path", os.path.join(self.model_name, "qry_encoder"),
                "--bsize", "1",
                "--index_location", os.path.join(self.output_dir, "index"),
                "--top_k", str(self.top_k),
            ]

            parser = HfArgumentParser(DPRSearchArguments)
            (dpr_args, remaining_args) = \
                parser.parse_args_into_dataclasses(return_remaining_strings=True, args=search_args)
            self.searcher = DPRSearcher(dpr_args)
        elif self._is_colbert:
            colbert_parser = Arguments(description='ColBERT indexing')

            colbert_parser.add_model_parameters()
            colbert_parser.add_model_inference_parameters()
            colbert_parser.add_indexing_input()
            colbert_parser.add_compressed_index_input()
            colbert_parser.add_argument('--nway', dest='nway', default=2, type=int)
            cargs = None
            index_args = [
                "prog",
                "--engine_type", "ColBERT",
                "--do_index",
                "--amp",
                "--bsize", "256",
                "--mask-punctuation",
                "--doc_maxlen", "180",
                "--model_name_or_path", self.model_name,
                "--index_name", os.path.join(self.output_dir, "index"),
                # "--root", self.colbert_root,
                "--nbits", "4",
                "--kmeans_niters", "20",
                "--collection", self.input_passages,
            ]

            # with patch.object(sys, 'argv', index_args):
            cargs = colbert_parser.parse(index_args)

            args_dict = vars(cargs)
            # remove keys not in ColBERTConfig
            args_dict = {key: args_dict[key] for key in args_dict if key not in ['run', 'nthreads', 'distributed', 'compression_level', 'input_arguments']}
            # args_dict to ColBERTConfig
            colBERTConfig = ColBERTConfig(**args_dict)

            with Run().context(RunConfig(root=cargs.root, experiment=cargs.experiment, nranks=cargs.nranks, amp=cargs.amp)):
                indexer = Indexer(cargs.checkpoint, colBERTConfig)
                indexer.index(name=cargs.index_name, collection=cargs.collection, overwrite=True)
            colbert_opts = [
                "prog",
                "--engine_type", "ColBERT",
                "--do_index",
                "--amp",
                "--bsize", "1",
                "--mask-punctuation",
                "--doc_maxlen", "180",
                "--model_name_or_path", self.model_name,
                "--index_location", os.path.join(self.output_dir, "index"),
                "--centroid_score_threshold", "0.4",
                "--ncells", "4",
                "--top_k", str(self.top_k),
                "--retrieve_only",
                "--ndocs", "40000",
                "--kmeans_niters", "20",
                "--collection", self.output_dir,
                # "--root", self.colbert_root,
                "--output_dir", self.output_dir,
                ]
            parser = Arguments(description='ColBERT search')

            parser.add_model_parameters()
            parser.add_model_inference_parameters()
            parser.add_compressed_index_input()
            parser.add_ranking_input()
            parser.add_retrieval_input()
            # search_args = parser.parse()
            with patch.object(sys, 'argv', colbert_opts):
                sargs = parser.parse()

            args_dict = vars(sargs)
            # remove keys not in ColBERTConfig
            args_dict = {key: args_dict[key] for key in args_dict if
                         key not in ['run', 'nthreads', 'distributed', 'compression_level', 'qrels', 'partitions',
                                     'retrieve_only', 'input_arguments']}
            colBERTConfig = ColBERTConfig(**args_dict)
            self.root = sargs.root
            self.experiment = sargs.experiment
            self.nranks=sargs.nargs
            self.amp=sargs.amp
            with Run().context(RunConfig(root=sargs.root, experiment=sargs.experiment, nranks=sargs.nranks, amp=sargs.amp)):
                self.searcher = Searcher(sargs.index_name, checkpoint=sargs.checkpoint, collection=sargs.collection,
                                    config=colBERTConfig)

                # rankings = searcher.search_all(args.queries, args.topK)
                # out_fn = os.path.join(args.output_dir, 'ranked_passages.tsv')
                # rankings.save(out_fn)
        else:
            raise RuntimeError("Unknown indexer type.")


    def __del__(self):
        self.tmp_dir.cleanup()

    def search(self, input_queries: List[AnyStr], batch_size=1, **kwargs):
        if self._is_dpr:
            return self._dpr_search(input_queries, batch_size, **kwargs)
        elif self._is_colbert:
            return self._colbert_search(input_queries, batch_size, **kwargs)
        else:
            print("Unknown indexer type.")
            raise RuntimeError("Unknown indexer type.")

    def _dpr_search(self, input_queries: List[AnyStr], batch_size=1, **kwargs):
        passage_ids = []
        scores = []

        batch = 0
        while batch<len(input_queries):
            batch_end = min(len(input_queries), batch+batch_size)
            # for query_number in tqdm(range(len(input_queries))):
            p_ids, response = self.searcher.search(
                query_batch=input_queries[batch: batch_end],
                top_k=self.top_k,
                mode="query_list"
            )
            passage_ids.extend(p_ids)
            scores.extend([r['scores'] for r in response])
            batch = batch_end

            # for rank, match in enumerate(p_ids[0]):
            #     out_ranks.append([input_queries[-1, match, rank + 1, response[0]['scores'][rank]])
        return passage_ids, scores

    def _colbert_search(self, input_queries: List[AnyStr], batch_size=1, **kwargs):
        passage_ids = []
        scores = []
        for query_number in tqdm(range(len(input_queries))):
            # for query_number in range(len(query_vectors)):
            p_ids, response = self.searcher.search_all(
                query_batch=[input_queries[query_number]],
                top_k=self.top_k
            )
            passage_ids.extend(p_ids[0])
            scores.extend(response[0]['scores'])
            # for rank, match in enumerate(q_ids[0]):
            #     out_ranks.append([input_queries[query_number]['id'], match, rank + 1, response[0]['scores'][rank]])
        return passage_ids, scores

    def encode(self, texts: Union[List[AnyStr], AnyStr], tokenizer, batch_size=64, **kwargs):
        if self._is_dpr:
            return self._dpr_encode(texts, tokenizer, batch_size, **kwargs)
        elif self._is_colbert:
            return self._colbert_encode(texts, tokenizer, batch_size, **kwargs)
        else:
            print("Unknown indexer type.")
            raise RuntimeError("Unknown indexer type.")

    def _dpr_encode(self, texts: Union[List[AnyStr], AnyStr], tokenizer, batch_size=64, **kwargs):
        if batch_size < 0:
            batch_size = self.batch_size
        if len(texts) > batch_size:
            embs = []
            for i in tqdm(range(0, len(texts), batch_size)):
                i_end = min(i + batch_size, len(texts))
                tems = queries_to_vectors(tokenizer, self.model, texts[i:i_end], max_query_length=500).tolist()
                embs.extend(tems)
        return embs
