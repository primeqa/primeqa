import os
import tempfile
from typing import List, AnyStr, Union
from unittest.mock import patch
import sys

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from primeqa.ir.dense.dpr_top.dpr.config import DPRIndexingArguments
#from primeqa.ir.dense.dpr_top.dpr.index_simple_corpus import DPRIndexer
from primeqa.ir.dense.dpr_top.dpr.searcher import DPRSearcher
from primeqa.ir.dense.dpr_top.dpr.config import DPRSearchArguments
from primeqa.ir.dense.colbert_top.colbert.indexer import Indexer
from primeqa.ir.dense.colbert_top.colbert.infra import Run, RunConfig, ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.utils.parser import Arguments
from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher
from primeqa.ir.dense.dpr_top.dpr.dpr_util import queries_to_vectors
from primeqa.components.indexer.dense import DPRIndexer
from primeqa.components.retriever.dense import DPRRetriever
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

    def __init__(self,
                 context_encoder_name_or_path=None,
                 query_encoder_name_or_path=None,
                 model_name=None,
                 batch_size=64,
                 top_k=10):
        """Creates a SearchableCorpus object from either a HuggingFace model id or a directory.
        It will automatically detect the index type (DPR, ColBERT, etc). Note: for DPR models, you need to
        define the context_encoder_name_or_path and query_encoder_name_or_path variables, for all other models
        you need to define the model_name parameter.
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
        self.ctx_encoder_name=context_encoder_name_or_path
        self.qry_encoder_name=query_encoder_name_or_path
        self.model_name=model_name
        if self.model_name is None:
            if self.ctx_encoder_name is None or self.qry_encoder_name is None:
                raise RuntimeError("Error: in SearchableCorpus.__init__, if model_name is None, then both "
                                   "context_encoder_name_or_path and query_encoder_name_or_path have to be defined!")
        self.output_dir = None
        self._is_dpr = True
        self.top_k = top_k
        
        # below code assumes HF model download has files in certain naming convention which right now is not valid.
        #ToDo: ColBERT- some part of the code below might get reintroduced later.

        # self.ctxt_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.model_name)
        
        # if not os.path.exists(model_name): # Assume a HF model name
        #     print("trying to download HF repo")
        #     model_name = hf_hub_download(repo_id=model_name, filename="config.json")
        # self.model_name = model_name
        # if os.path.exists(os.path.join(model_name,"ctx_encoder")): # Looks like a DPR model
        #     self._is_dpr = True
        # else:
        #     self._is_colbert = True
        # self.ctxt_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        #     os.path.join(self.model_name,"ctx_encoder"))
        self.tmp_dir = None
        self.indexer = None
        self.searcher = None
        self.input_passages = None
        self.working_dir = None


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
        if type(texts)==str:
            self.input_passages = texts
        else:
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
            index_args = [
                "prog",
                "--bsize", "16",
                "--ctx_encoder_name_or_path", os.path.join(self.ctx_encoder_name, "ctx_encoder"),
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
                "--model_name_or_path", os.path.join(self.qry_encoder_name, "qry_encoder"),
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

            with patch.object(sys, 'argv', index_args):
                cargs = colbert_parser.parse()

            args_dict = vars(cargs)
            # remove keys not in ColBERTConfig
            args_dict = {key: args_dict[key] for key in args_dict if key not in ['run', 'nthreads', 'distributed', 'compression_level', 'input_arguments']}
            # args_dict to ColBERTConfig
            colBERTConfig: ColBERTConfig = ColBERTConfig(**args_dict)

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


    def add_documents(self,input_file):
        # doc_class = DocumentCollection(input_file)
        # self.tmp_dir = tempfile.TemporaryDirectory()
        # self.working_dir = self.tmp_dir.name
        # os.makedirs(os.path.join(self.working_dir, "processed_data"))
        # out_file= os.path.join(self.working_dir, "processed_data","processed_input.tsv")
        # output_file_path= doc_class.write_corpus_tsv(out_file)
        dpr = DPRIndexer(ctx_encoder_model_name_or_path=self.ctx_encoder_name, vector_db="FAISS")
        dpr.index(input_file)

        self.searcher = DPRRetriever(index_root=dpr.index_root,
                                     query_encoder_model_name_or_path=self.qry_encoder_name,
                                     indexer=dpr,
                                     index_name=dpr.index_name,
                                     max_num_documents=self.top_k)
        #self.searcher=self.searcher.get_searcher()


        

    def select_column(data, col):
        return [d[col] for d in data]

    def __del__(self):
        self.tmp_dir.cleanup()

    def search(self, input_queries: List[AnyStr], batch_size=1, **kwargs):
        return self.searcher.predict(input_queries,return_passages=True)

    def search_not_in_use(self, input_queries: List[AnyStr], batch_size=1, **kwargs):
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
            p_ids, response = self.searcher.search(
                query_batch=input_queries[batch: batch_end],
                top_k=self.top_k,
                mode="query_list"
            )
            passage_ids.extend(p_ids)
            scores.extend([r['scores'] for r in response])
            batch = batch_end

        return passage_ids, scores

    def _colbert_search(self, input_queries: List[AnyStr], batch_size=1, **kwargs):
        passage_ids = []
        scores = []
        for query_number in tqdm(range(len(input_queries))):
            p_ids, response = self.searcher.search_all(
                query_batch=[input_queries[query_number]],
                top_k=self.top_k
            )
            passage_ids.extend(p_ids[0])
            scores.extend(response[0]['scores'])
        return passage_ids, scores

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
            itm = {'text': (row["title"] + ' '  if 'title' in row else '') + row["text"],
                             'id': row['id']}
            if 'title'in row:
                itm['title'] = row['title']
            if 'relevant' in row:
                itm['relevant'] = row['relevant']
            if 'answers' in row and row['answers'] is not None:
                itm['answers'] = row['answers'].split("::")
            passages.append(itm)
    return passages

def compute_score(input_queries, input_passages, answers,  ranks=[1,3,5,10], verbose=False):
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
        gt[q['id']] = {id: 1 for id in q['relevant'].split(" ") if int(id) >= 0}

    def skip(out_ranks, record, rid):
        qid = record[0]
        while rid < len(out_ranks) and out_ranks[rid][0] == qid:
            rid += 1
        return rid

    def update_scores(ranks, rnk, scores):
        j = 0
        while ranks[j] < rnk:
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

    for qi, q in enumerate(input_queries):
        tmp_scores = {r: 0 for r in ranks}
        tmp_lscores = {r: 0 for r in ranks}
        qid = input_queries[qi]['id']
        outranks = []
        for ai, ans in enumerate(answers[qi]):
            if verbose:
                outr = [qid, ans, ai+1]
            if str(ans) in gt[qid]:  # Great, we found a match.
                update_scores(ranks, ai+1, tmp_scores)
                # break
                if verbose:
                    outr.append(1)
            elif verbose:
                outr.append(0)
            if verbose:
                outranks.append(outr)

        if with_answers:
            for ai, ans in enumerate(answers[qi]):
                inputq = input_queries[qi]
                txt = input_passages[rp_map[ans]]['text'].lower()
                found = False
                for s in inputq['answers']:
                    if txt.find(s.lower()) >= 0:
                        found = True
                        break
                if (found):
                    update_scores(ranks, ai+1, tmp_lscores)
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
                out.write("\t".join([str(s) for s in entry])+"\n")
    return res