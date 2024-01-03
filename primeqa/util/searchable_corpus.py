import os
import shutil
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
from primeqa.ir.dense.colbert_top.colbert.infra import Run, RunConfig, ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.utils.parser import Arguments
from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher
from primeqa.ir.dense.dpr_top.dpr.dpr_util import queries_to_vectors
from primeqa.ir.sparse.retriever import PyseriniRetriever
from primeqa.ir.sparse.indexer import PyseriniIndexer
import json
import numpy as np

from transformers import (
    HfArgumentParser,
    DPRContextEncoderTokenizer
)


class SearchableCorpus:
    """Class that allows easy indexing and searching of documents. It's meant to be used in the
    context of the RAG pattern (retrive-and-generate). It provides 2 main apis:
    * SearchableCorpus.add(List[Tuple[text:AnyStr, title:AnyStr, id:AnyStr]]) - adds documents to the collection
    * SearchableCorpus.search(queries: List[AnyStr]) - retrieves documents that are relevant to a set of queries

    It currently wraps the DPR index, and it will support ColBERT with the same interface soon."""
    _UnknownModelType = 0
    _DPR = 1
    _ColBERT = 2
    _BM25 = 3
    _ES_BM25 = 4
    _ES_DENSE = 5
    _ES_ELSER = 6

    def __init__(self, model_name, batch_size=64, top_k=10, **kwargs):
        """Creates a SearchableCorpus object from either a HuggingFace model id or a directory.
        It will automatically detect the index type (DPR, ColBERT, etc).
        Args:
            :param model_name: AnyStr -
                 defines the model - it should either be a HuggingFace id or a directory
            :param context_encoder_name_or_path: AnyStr - defines the context encoder. For DPR only.
            :param query_encoder_name_or_path: AnyStr - defines the query encoder. For DPR only.
            :param batch_size: int
                 defines the ingestion batch size.
            :param top_k: int
                 - defines the default number of retrieved answers. It can be changed in .search()
            """
        self.amp = None
        self.nranks = None
        self.experiment = None
        self.root = None
        self.output_dir = None
        self._model_type = SearchableCorpus._UnknownModelType
        self.top_k = top_k
        self.batch_size = batch_size

        self.model_name = model_name
        if model_name == "bm25":
            self._model_type = SearchableCorpus._BM25
        elif model_name == "es_bm25":
            self._model_type = SearchableCorpus._ES_BM25
        elif model_name == "es_dense":
            self._model_type = SearchableCorpus._ES_DENSE
        elif model_name == "es_elser":
            self.model_name = SearchableCorpus._ES_ELSER
        elif not os.path.exists(model_name):  # Assume a HF model name
            model_name = hf_hub_download(repo_id=model_name, filename="config.json")
        else:
            if os.path.exists(os.path.join(model_name, "ctx_encoder")):  # Looks like a DPR model
                self._model_type = SearchableCorpus._DPR
                from transformers import AutoTokenizer
                self.ctxt_tokenizer = AutoTokenizer.from_pretrained(
                    os.path.join(self.model_name, "ctx_encoder"))
            else:
                self._model_type = SearchableCorpus._ColBERT

        if model_name.startswith("es_"):
            self._init_ES_settings()

        self.tmp_dir = None
        self.indexer = None
        self.searcher = None
        self.input_passages = None
        self.working_dir = None
        if 'colbert_index' in kwargs:
            self._colbert_index = kwargs['colbert_index']
        else:
            self._colbert_index = None
        if model_name in ['es_bm25', 'es_dense', 'es_elser']:
            from elasticsearch import Elasticsearch
            self.index_name = SearchableCorpus._get_args(kwargs, 'index_name')
            self.fields = SearchableCorpus._get_args(kwargs, 'fields')
            if self.fields is None:
                # raise RuntimeError(f"For elasticsearch, you need to provide the search fields.")
                print(f"The 'fields' param not defined, using ['text']")
                self.fields=['text']
            server = SearchableCorpus._get_args(kwargs, 'server')
            ssl_fingerprint = SearchableCorpus._get_args(kwargs,
                                                 'ES_SSL_FINGERPRINT',
                                                 os.getenv("ES_SSL_FINGERPRINT"))
            ssl_api_key = SearchableCorpus._get_args(kwargs,
                                                     'ES_API_KEY',
                                                     os.getenv('ES_API_KEY'))
            es_passwd = SearchableCorpus._get_args(kwargs,
                                                   'ES_PASSWORD',
                                                   os.getenv('ES_PASSWORD'))
            cloud_id = SearchableCorpus._get_args(kwargs, 'cloud_id')
            if not server:
                print(f"No server provided for model {model_name}")
                raise RuntimeError(f"No server provided for model {model_name}")
            if ssl_fingerprint and ssl_api_key:
                self.client = Elasticsearch(self.server,
                                            ssl_assert_fingerprint= ssl_fingerprint,
                                            api_key=ssl_api_key)
            elif es_passwd and cloud_id:
                self.client = Elasticsearch(cloud_id=cloud_id, basic_auth=("elastic", es_passwd))
            else:
                print(f"No credentials provided for server {server}, model_name {model_name}")

    @staticmethod
    def _get_args(_dict, _val, _default=None):
        return _dict[_val] if _val in _dict else _default

    def add(self, texts:Union[AnyStr, List[AnyStr]], titles:List[AnyStr]=None, ids:List[AnyStr]=None, **kwargs):
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
        if type(texts) == str:
            self.input_passages = texts
        else:
            os.makedirs(os.path.join(self.working_dir, "input_dir"))
            self.input_passages = os.path.join(self.working_dir, "input_dir", "input.tsv")
            with open(self.input_passages, "w") as w:
                w.write("\t".join(['id', 'text', 'title']) + "\n")
                for i, t in enumerate(texts):
                    w.write("\t".join([
                        str(i + 1) if ids is None else ids[i],
                        texts[i].strip(),
                        titles[i] if titles is not None else ""
                    ]) + "\n"
                            )
        if self._model_type == SearchableCorpus._DPR:
            index_args = [
                "prog",
                "--bsize", "16",
                "--ctx_encoder_name_or_path", os.path.join(self.model_name, "ctx_encoder"),
                "--embed", "1of1",
                "--output_dir", os.path.join(self.output_dir, "index"),
                "--collection", self.input_passages,
            ]

            parser = HfArgumentParser(DPRIndexingArguments)
            (dpr_args, remaining_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True,
                                                                            args=index_args)

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
        elif self._model_type == SearchableCorpus._ColBERT:
            if self._colbert_index is None:
                colbert_parser = Arguments(description='ColBERT indexing')

                colbert_parser.add_model_parameters()
                colbert_parser.add_model_inference_parameters()
                colbert_parser.add_indexing_input()
                colbert_parser.add_compressed_index_input()
                colbert_parser.add_argument('--nway', dest='nway', default=2, type=int)
                cargs = None
                index_args = [
                    "prog",
                    "--amp",
                    "--bsize", "256",
                    "--mask-punctuation",
                    "--doc_maxlen", "180",
                    "--model_name_or_path", self.model_name,
                    "--index_name", os.path.join(self.output_dir, "index"),
                    "--root", os.path.join(os.path.join(self.output_dir, "root")),
                    "--nbits", "4",
                    "--kmeans_niters", "20",
                    "--collection", self.input_passages,
                ]

                with patch.object(sys, 'argv', index_args):
                    cargs = colbert_parser.parse()
                    # cargs = colbert_parser.parse(index_args)

                args_dict = vars(cargs)
                # remove keys not in ColBERTConfig
                args_dict = {key: args_dict[key] for key in args_dict if
                             key not in ['run', 'nthreads', 'distributed', 'compression_level', 'input_arguments']}
                # args_dict to ColBERTConfig
                colBERTConfig: ColBERTConfig = ColBERTConfig(**args_dict)

                with Run().context(
                        RunConfig(root=cargs.root, experiment=cargs.experiment, nranks=cargs.nranks, amp=cargs.amp)):
                    indexer = Indexer(cargs.checkpoint, colBERTConfig)
                    indexer.index(name=cargs.index_name, collection=cargs.collection, overwrite=True)
                    shutil.copytree(cargs.index_name, os.path.join(os.path.dirname(self.model_name), "index"))
                self._colbert_index = cargs.index_name

            colbert_opts = [
                "prog",
                "--amp",
                "--bsize", "1",
                "--mask-punctuation",
                "--doc_maxlen", "180",
                "--model_name_or_path", self.model_name,
                "--index_location", self._colbert_index,
                "--centroid_score_threshold", "0.4",
                "--ncells", "4",
                "--top_k", str(self.top_k),
                "--ndocs", "40000",
                "--kmeans_niters", "20",
                "--collection", self.output_dir,
                "--root", os.path.join(os.path.join(self.output_dir, "root")),
                "--output_dir", self.output_dir,
            ]
            parser = Arguments(description='ColBERT search')

            parser.add_model_parameters()
            parser.add_model_inference_parameters()
            parser.add_compressed_index_input()
            parser.add_ranking_input()
            parser.add_retrieval_input()
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
            self.nranks = sargs.nranks
            self.amp = sargs.amp

            with Run().context(RunConfig(root=self.root, experiment=self.experiment, nranks=self.nranks, amp=self.amp)):
                self.searcher = Searcher(sargs.index_name, checkpoint=sargs.checkpoint, collection=sargs.collection,
                                         config=colBERTConfig)

                # rankings = searcher.search_all(args.queries, args.topK)
                # out_fn = os.path.join(args.output_dir, 'ranked_passages.tsv')
                # rankings.save(out_fn)
        elif self._model_type == SearchableCorpus._BM25:
            indexer = PyseriniIndexer()
            index_path = os.path.join(self.output_dir, 'index')
            indexer.index_collection(collection=self.input_passages,
                                     index_path=index_path,
                                     threads=10)
            self.searcher = PyseriniRetriever(index_location=index_path, use_bm25=True)
        else:
            raise RuntimeError("Unknown indexer type.")

    def __del__(self):
        self.tmp_dir.cleanup()

    def _init_ES_settings(self):
        self.standard_mappings = {
            "properties": {
                "ml.tokens": {
                    "type": "rank_features"
                },
                "title": {"type": "text", "analyzer": "english"},
                "text": {"type": "text", "analyzer": "english"},
                "url": {"type": "text", "analyzer": "english"},
            }
        }
        self.settings = {
            "number_of_replicas": 0,
            "number_of_shards": 1,
            "refresh_interval": "1m",
            "analysis": {
                "filter": {
                    "possessive_english_stemmer": {
                        "type": "stemmer",
                        "language": "possessive_english"
                    },
                    "light_english_stemmer": {
                        "type": "stemmer",
                        "language": "light_english"
                    },
                    "english_stop": {
                        "ignore_case": "true",
                        "type": "stop",
                        "stopwords": ["a", "about", "all", "also", "am", "an", "and", "any", "are", "as", "at",
                                      "be", "been", "but", "by", "can", "de", "did", "do", "does", "for", "from",
                                      "had", "has", "have", "he", "her", "him", "his", "how", "if", "in", "into",
                                      "is", "it", "its", "more", "my", "nbsp", "new", "no", "non", "not", "of",
                                      "on", "one", "or", "other", "our", "she", "so", "some", "such", "than",
                                      "that", "the", "their", "then", "there", "these", "they", "this", "those",
                                      "thus", "to", "up", "us", "use", "was", "we", "were", "what", "when", "where",
                                      "which", "while", "why", "will", "with", "would", "you", "your", "yours"]
                    }
                },
                "analyzer": {
                    "text_en_no_stop": {
                        "filter": [
                            "lowercase",
                            "possessive_english_stemmer",
                            "light_english_stemmer"
                        ],
                        "tokenizer": "standard"
                    },
                    "text_en_stop": {
                        "filter": [
                            "lowercase",
                            "possessive_english_stemmer",
                            "english_stop",
                            "light_english_stemmer"
                        ],
                        "tokenizer": "standard"
                    },
                    "whitespace_lowercase": {
                        "tokenizer": "whitespace",
                        "filter": [
                            "lowercase"
                        ]
                    }
                },
                "normalizer": {
                    "keyword_lowercase": {
                        "filter": [
                            "lowercase"
                        ]
                    }
                }
            }
        }
        self.coga_mappings = {
            "_source": {
                "enabled": "true"
            },
            "dynamic": "false",
            "properties": {
                "url": {
                    "type": "text"
                },
                "title": {
                    "type": "text",
                    "analyzer": "text_en_no_stop",
                    "search_analyzer": "text_en_stop",
                    "term_vector": "with_positions_offsets",
                    "index_options": "offsets",
                    "store": "true"
                },
                "fileTitle": {
                    "type": "text",
                    "analyzer": "text_en_no_stop",
                    "search_analyzer": "text_en_stop",
                    "term_vector": "with_positions_offsets",
                    "index_options": "offsets",
                    "store": "true"
                },
                "title_paraphrases": {
                    "type": "text",
                    "analyzer": "text_en_no_stop",
                    "search_analyzer": "text_en_stop",
                    "term_vector": "with_positions_offsets",
                    "index_options": "offsets",
                    "store": "true"
                },
                "productId": {
                    "type": "keyword"
                },
                "deliverableLoio": {
                    "type": "keyword",
                },
                "filePath": {
                    "type": "keyword",
                },
                "text": {
                    "type": "text",
                    "analyzer": "text_en_no_stop",
                    "search_analyzer": "text_en_stop",
                    "term_vector": "with_positions_offsets",
                    "index_options": "offsets",
                    "store": "true"
                },
                "plainTextContent": {
                    "type": "text",
                    "analyzer": "text_en_no_stop",
                    "search_analyzer": "text_en_stop",
                    "term_vector": "with_positions_offsets",
                    "index_options": "offsets",
                    "store": "true"
                },
                "title_and_text": {
                    "type": "text",
                    "analyzer": "text_en_no_stop",
                    "search_analyzer": "text_en_stop",
                    "term_vector": "with_positions_offsets",
                    "index_options": "offsets",
                    "store": "true"
                },
                "app_name": {
                    "type": "text",
                    "analyzer": "text_en_no_stop",
                    "search_analyzer": "text_en_stop",
                    "term_vector": "with_positions_offsets",
                    "index_options": "offsets",
                    "store": "true"
                },
                "collection": {
                    "type": "text",
                    "fields": {
                        "exact": {
                            "normalizer": "keyword_lowercase",
                            "type": "keyword",
                            "doc_values": "false"
                        }
                    }
                }}
        }

    def search(self, input_queries: List[AnyStr], batch_size=1, **kwargs):
        """Retrieves the most relevant documents in the collection for a given set of queries.
        Args:
            * input_queries: List[AnyStr]
               - the list of input queries (strings)
            * batch_size: int, default 1
               - defines the number of documents to return for each question. Default is 1.
            * kwargs might contain additional arguments going foward.
        Returns:
            Tuple[List[List[AnyStr]], List[List[Float]]]
            - is a list of document IDs per query and an associated list of scores per query, for all the queries.
            """
        call_funcs = {SearchableCorpus._DPR: self._dpr_search,
                      SearchableCorpus._ColBERT: self._colbert_search,
                      SearchableCorpus._BM25: self._bm25_search,
                      SearchableCorpus._ES_BM25: self._es_bm25_search,
                      SearchableCorpus._ES_DENSE: self._es_dense_search,
                      SearchableCorpus._ES_ELSER: self._es_elser_search}
        if self._model_type not in call_funcs:
            print("Unknown indexer type.")
            raise RuntimeError("Unknown indexer type.")
        else:
            res = call_funcs[self._model_type](input_queries, batch_size, **kwargs)
            if SearchableCorpus._get_args(kwargs, 'get_text'):
                return res
            else:
                return res[0], res[1]

    # if self._model_type == SearchableCorpus._DPR:
    #     return self._dpr_search(input_queries, batch_size, **kwargs)
    # elif self._model_type == SearchableCorpus._ColBERT:
    #     return self._colbert_search(input_queries, batch_size, **kwargs)
    # else:

    def _dpr_search(self, input_queries: List[AnyStr], batch_size=1, **kwargs):
        """Retrieves the most relevant documents in the collection for a given set of queries.
        Args:
            * input_queries: List[AnyStr]
               - the list of input queries (strings)
            * batch_size: int, default 1
               - defines the number of documents to return for each question. Default is 1.
            * kwargs might contain additional arguments going foward.
        Returns:
            Tuple[List[List[AnyStr]], List[List[Float]]]
            - is a list of document IDs per query and an associated list of scores per query, for all the queries.
            """
        passage_ids = []
        scores = []
        texts = []

        batch = 0
        while batch < len(input_queries):
            batch_end = min(len(input_queries), batch + batch_size)
            p_ids, response = self.searcher.search(
                query_batch=input_queries[batch: batch_end],
                top_k=self.top_k,
                mode="query_list"
            )
            passage_ids.extend(p_ids)
            scores.extend([r['scores'] for r in response])
            texts.extend([t['text'] for t in response])
            batch = batch_end

        return passage_ids, scores

    def _colbert_search(self, input_queries: List[AnyStr], batch_size=1, **kwargs):
        """Retrieves the most relevant documents in the collection for a given set of queries using ColBERT.
        Args:
            * input_queries: List[AnyStr]
               - the list of input queries (strings)
            * batch_size: int, default 1
               - defines the number of documents to return for each question. Default is 1.
            * kwargs might contain additional arguments going foward.
        Returns:
            Tuple[List[List[AnyStr]], List[List[Float]]]
            - is a list of document IDs per query and an associated list of scores per query, for all the queries.
            """
        passage_ids = []
        scores = []
        with Run().context(
                RunConfig(root=self.root,
                          experiment=self.experiment,
                          nranks=self.nranks,
                          amp=self.amp)):
            for query_number in range(len(input_queries)):
                p_ids, ranks, scrs = self.searcher.search(
                    text=input_queries[query_number],
                    k=self.top_k
                )
                passage_ids.append([str(p) for p in p_ids])
                scores.append(scrs)
        return passage_ids, scores, []

    def _bm25_search(self, input_queries: List[AnyStr], batch_size=1, **kwargs):
        """Retrieves the most relevant documents in the collection for a given set of queries
           using pyserini BM25 search.
        Args:
            * input_queries: List[AnyStr]
               - the list of input queries (strings)
            * batch_size: int, default 1
               - defines the number of documents to return for each question. Default is 1.
            * kwargs might contain additional arguments going foward.
        Returns:
            Tuple[List[List[AnyStr]], List[List[Float]]]
            - is a list of document IDs per query and an associated list of scores per query, for all the queries.
            """
        passage_ids = []
        scores = []
        texts = []
        for query_number, q in enumerate(input_queries):
            res = self.searcher.retrieve(query=q, topK=self.top_k)
            passage_ids.append([r['doc_id'] for r in res])
            scores.append([r['score'] for r in res])
            scores.append([r['text'] for r in res])
        return passage_ids, scores, texts

    def _es_bm25_search(self, input_queries: List[AnyStr], batch_size=1, **kwargs):
        """Retrieves the most relevant documents in the collection for a given set of queries using ElasticSearch
           BM25.
        Args:
            * input_queries: List[AnyStr]
               - the list of input queries (strings)
            * batch_size: int, default 1
               - defines the number of documents to return for each question. Default is 1.
            * kwargs might contain additional arguments going foward.
        Returns:
            Tuple[List[List[AnyStr]], List[List[Float]]]
            - is a list of document IDs per query and an associated list of scores per query, for all the queries.
            """
        passage_ids = []
        scores = []
        texts = []
        for query_number, q in enumerate(input_queries):
            es_query = {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": q,
                            "fields": self.fields
                        }
                    },
                }
            }
            query = {
                "size": self.top_k,
                "query": es_query
            }
            res = self.client.search(index=self.index_name, query=query)
            self._es_extract_results(res, passage_ids, scores, texts)
        return passage_ids, scores, texts

    def _es_extract_results(self, res, passage_ids, scores, texts):
        """Extracts the passage id, score and text out of the ElasticSearch response and adds the result to the
        previous lists.
        Args:
            * res: ElasticSearch response
            * passage_ids: List[str] - the list of passage ids so far
            * scores: List[float] - the list of scores so far.
            * texts: List[float]: the list of texts so far.
        Returns:
            nothing """
        for rank, r in enumerate(res.body['hits']['hits']):
            passage_ids.append(r['_id'])
            scores.append(r['_score'])
            texts.append(r['_source']['text'])

    def _es_dense_search(self, input_queries: List[AnyStr], batch_size=1, **kwargs):
        """Retrieves the most relevant documents in the collection for a given set of queries using Elasticsearch
           dense search.
        Args:
            * input_queries: List[AnyStr]
               - the list of input queries (strings)
            * batch_size: int, default 1
               - defines the number of documents to return for each question. Default is 1.
            * kwargs might contain additional arguments going foward.
        Returns:
            Tuple[List[List[AnyStr]], List[List[Float]]]
            - is a list of document IDs per query and an associated list of scores per query, for all the queries.
            """
        passage_ids = []
        scores = []
        texts = []
        for query_number in tqdm(range(len(input_queries))):
            query_vector = self._encode(self.model, input_queries[query_number]['text'],
                                   normalize_embs=self._get_args(kwargs, 'normalize_embs', False))
            qid = input_queries[query_number]['id']
            query = {
                "field": "vector",
                "query_vector": query_vector,
                "k": self.top_k,
                "num_candidates": 1000,
            }
            res = self.client.search(
                index=self.index_name,
                knn=query,
                size=self.top_k,
                source_excludes=['vector']
            )
            self._es_extract_results(res, passage_ids, scores, texts)
        return passage_ids, scores, texts


    def _es_elser_search(self, input_queries: List[AnyStr], batch_size=1, **kwargs):
        """Retrieves the most relevant documents in the collection for a given set of queries using ES ELSER search.
        Args:
            * input_queries: List[AnyStr]
               - the list of input queries (strings)
            * batch_size: int, default 1
               - defines the number of documents to return for each question. Default is 1.
            * kwargs might contain additional arguments going foward.
        Returns:
            Tuple[List[List[AnyStr]], List[List[Float]]]
            - is a list of document IDs per query and an associated list of scores per query, for all the queries.
            """
        passage_ids = []
        scores = []
        texts = []
        for query_number in tqdm(range(len(input_queries))):
            qid = input_queries[query_number]['id']
            query = {
                "text_expansion": {
                    "ml.tokens": {
                        "model_id": ".elser_model_1",
                        "model_text": input_queries[query_number]['text']
                    }
                }
            }
            res = self.client.search(
                index=self.index_name,
                source_excludes=['ml.tokens'],
                size=self.top_k,
            )
            self._es_extract_results(res, passage_ids, scores, texts)
        return passage_ids, scores, texts


    def encode(self, texts: Union[List[AnyStr], AnyStr], tokenizer, batch_size=64, **kwargs):
        """ Encodes a list of context documents, returning their dense representation.
        It will be used when the code will no longer insist in writing everything to disk.
        Arguments:
            * texts: Union[List[AnyStr], AnyStr]
              - either a list of documents or just a single document
            * tokenizer: Tokenizer
              - The tokenizer used to transform the strings into word-pieces, then integers
            * batch_size: int
              - The batch size used in indexing.
        Returns:
            The list of embeddings (or the one embedding) for the given documents (document).
            """
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

    def _encode(self, texts: Union[List[AnyStr], AnyStr], tokenizer, batch_size=64, **kwargs):
        query_vector = self.model.encode(texts)
        if 'normalize_embs' in kwargs and kwargs['normalize_embs']:
            query_vector = self._normalize([query_vector])[0]
        return query_vector

    def _normalize(passage_vectors):
        return [v / np.linalg.norm(v) for v in passage_vectors if np.linalg.norm(v) > 0]

def read_tsv_data(input_file, fields=None):
    """
    Utility function to read the standard tuple representation for both passages and questions.
    Args:
        input_file: the tsv file with either contexts or queries, tab-separated.
        fields: the schema of the file. It can be either ['id', 'text', 'title'] for contexts,
        ['id', 'text', 'relevant', 'answers'] or ['id', 'text', 'relevant'] for questions

    Returns: List[Dict[AnyStr: AnyStr]]
       the list of tuples

    """
    import csv
    passages = []
    if fields is None:
        num_args = 3
    else:
        num_args = len(fields)
    with open(input_file) as in_file:
        csv_reader = \
            csv.DictReader(in_file, fieldnames=fields, delimiter="\t") \
                if fields is not None \
                else csv.DictReader(in_file, delimiter="\t")
        next(csv_reader)
        for row in csv_reader:
            assert len(row) in [2, 3, 4], f'Invalid .tsv record (has to contain 2 or 3 fields): {row}'
            itm = {'text': (row["title"] + ' ' if 'title' in row else '') + row["text"],
                   'id': row['id']}
            if 'title' in row:
                itm['title'] = row['title']
            if 'relevant' in row:
                itm['relevant'] = row['relevant']
            if 'answers' in row and row['answers'] is not None:
                itm['answers'] = row['answers'].split("::")
            passages.append(itm)
    return passages


def compute_score(input_queries, input_passages, answers, ranks=[1, 3, 5, 10], verbose=False):
    """
    Computes the success at different levels of recall, given the goldstandard passage indexes per query.
    It computes two scores:
       * Success at rank_i, defined as sum_q 1_{top i answers for question q contains a goldstandard passage} / #questions
       * Lenient success at rank i, defined as
                sum_q 1_{ one in the documents in top i for question q contains a goldstandard answer) / #questions
    Note that a document that contains the actual textual answer does not necesarily answer the question, hence it's a
    more lenient evaluation. Any goldstandard passage will contain a goldstandard answer text, by definition.
    Args:
        input_queries: List[Dict['id': AnyStr, 'text': AnyStr, 'relevant': AnyStr, 'answers': AnyStr]]
           - the input queries. Each query is a dictionary with the keys 'id','text', 'relevant', 'answers'.
        input_passages: List[Dict['id': AnyStr, 'text': AnyStr', 'title': AnyStr]]
           - the input passages. These are used to create a reverse-index list for the passages (so we can get the
             text for a given passage ID)
        answers: List[List[AnyStr]]
           - the retrieved passages IDs for each query
        ranks: List[int]
           - the ranks at which to compute success
        verbose: Bool
           - Will save a file with the individual (query, passage_answer) tuples.

    Returns:

    """
    if "relevant" not in input_queries[0] or input_queries[0]['relevant'] is None:
        print("The input question file does not contain answers. Please fix that and restart.")
        sys.exit(12)

    scores = {r: 0 for r in ranks}
    lscores = {r: 0 for r in ranks}

    gt = {}
    for q in input_queries:
        gt[q['id']] = {id: 1 for id in q['relevant'].split(",") if int(id) >= 0}

    def skip(out_ranks, record, rid):
        qid = record[0]
        while rid < len(out_ranks) and out_ranks[rid][0] == qid:
            rid += 1
        return rid

    def update_scores(ranks, rnk, scores):
        j = 0
        while j<len(ranks) and ranks[j] < rnk:
            j += 1
        for k in ranks[j:]:
            scores[k] += 1

    def reverse_map(input_queries):
        rq_map = {}
        for i, q in enumerate(input_queries):
            rq_map[q['id']] = i
        return rq_map

    with_answers = False
    if 'answers' in input_queries[0]:
        with_answers = True
    rp_map = reverse_map(input_passages)
    if verbose:
        out_result = []

    for qi, q in tqdm(enumerate(input_queries)):
        # print(f"Processing query {qi}")
        tmp_scores = {r: 0 for r in ranks}
        tmp_lscores = {r: 0 for r in ranks}
        qid = input_queries[qi]['id']
        outranks = []
        for ai, ans in enumerate(answers[qi]):
            if verbose:
                outr = [qid, ans, ai + 1]
            if str(ans) in gt[qid]:  # Great, we found a match.
                update_scores(ranks, ai + 1, tmp_scores)
                # break
                if verbose:
                    outr.append(1)
            elif verbose:
                outr.append(0)
            if verbose:
                outranks.append(outr)

        if with_answers:
            inputq = input_queries[qi]
            text_answers = inputq['answers']
            for ai, ans in enumerate(answers[qi]):
                if ans not in rp_map:
                    print(f"On question {qi}, the answer {ans} is not in the reverse passage map.")
                txt = input_passages[rp_map[ans]]['text'].lower()
                found = False
                # for textans in text_answers:
                #     if txt.find(textans.lower()) >= 0:
                #         found = True
                #         break
                txt_ans = text_answers[0]
                found = txt.find(txt_ans.lower())>=1
                if (found):
                    update_scores(ranks, ai + 1, tmp_lscores)
                if verbose:
                    outranks[ai].append(int(found))
        if verbose:
            out_result.extend(outranks)

        for r in ranks:
            scores[r] += int(tmp_scores[r] >= 1)
            lscores[r] += int(tmp_lscores[r] >= 1)

    res = {"num_ranked_queries": len(input_queries),
           "num_judged_queries": len(input_queries),
           "success":
               {r: int(1000 * scores[r] / len(input_queries)) / 1000.0 for r in ranks},
           "lenient_success":
               {r: int(1000 * lscores[r] / len(input_queries)) / 1000.0 for r in ranks}
           }
    if verbose:
        with open("result.augmented", "w") as out:
            for entry in out_result:
                out.write("\t".join([str(s) for s in entry]) + "\n")
    return res