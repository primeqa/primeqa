import os, re, json, csv
import logging
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from typing import List, Union
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import logging
import sys
import pyizumo


nlp = None

def old_split_passages(text: str, tokenizer, max_length: int = 512, stride: int = None) \
        -> List[str]:
    """
    Method to split a text into pieces that are of a specified <max_length> length, with the
    <stride> overlap, using a HF tokenizer.
    :param text: str - the text to split
    :param tokenizer: HF Tokenizer
       - the tokenizer to do the work of splitting the words into word pieces
    :param max_length: int - the maximum length of the resulting sequence
    :param stride: int - the overlap between windows
    """
    text = re.sub(r' {2,}', ' ', text, flags=re.MULTILINE)  # remove multiple spaces.
    if max_length is not None:
        res = tokenizer(text, max_length=max_length, stride=stride,
                        return_overflowing_tokens=True, truncation=True)
        if len(res['input_ids']) == 1:
            return [text]
        else:
            texts = []
            end = re.compile(f' {re.escape(tokenizer.sep_token)}$')
            for split_passage in res['input_ids']:
                tt = end.sub(
                    "",
                    tokenizer.decode(split_passage).replace(f"{tokenizer.cls_token} ", "")
                )
                texts.append(tt)
            return texts

def split_text(text: str, tokenizer, max_length: int = 512, stride: int = None) \
        -> List[str]:
    """
    Method to split a text into pieces that are of a specified <max_length> length, with the
    <stride> overlap, using a HF tokenizer.
    :param text: str - the text to split
    :param tokenizer: HF Tokenizer
       - the tokenizer to do the work of splitting the words into word pieces
    :param max_length: int - the maximum length of the resulting sequence
    :param stride: int - the overlap between windows
    """
    text = re.sub(r' {2,}', ' ', text, flags=re.MULTILINE)  # remove multiple spaces.
    if max_length is not None:
        res = tokenizer(text, max_length=max_length, stride=stride,
                        return_overflowing_tokens=True, truncation=True)
        if len(res['input_ids']) == 1:
            return [text]
        else:
            if not nlp:
                nlp = pyizumo.load("en")
            texts = []
            parsed_text = nlp(text)

            start = 0
            end = 0
            length = 0
            prev_end = -1
            prev_start = -1
            prev_len = -1
            positions=[]
            for sent in parsed_text.sentences:
                stext = sent.text
                res = tokenizer(stext, max_length=max_length, stride=stride,
                        return_overflowing_tokens=True, truncation=True)
                slen = len(res['input_ids'])
                if length + slen < max_length:
                    length += slen
                    prev_end = sent.end
                else:
                    if length > 0: # There is at least something in the list
                        texts.append(text[start:prev_end])
                        positions.append([start, prev_end])
                    if slen > max_length:
                        print("We have a problem: {text}")
                        continue
                    start = sent.begin
                    length = slen
            if length>0:
                texts.append(text[start:prev_end])
                positions.append([start, prev_end])

            # end = re.compile(f' {re.escape(tokenizer.sep_token)}$')

            # for split_passage in res['input_ids']:
            #     tt = end.sub(
            #         "",
            #         tokenizer.decode(split_passage).replace(f"{tokenizer.cls_token} ", "")
            #     )
            #     texts.append(tt)
            return texts, positions


def get_tokenized_length(tokenizer, text):
    if tokenizer is not None:
        toks = tokenizer(text)
        return len(toks['input_ids'])
    else:
        return -1


def read_data(input_file, fields=None, remove_url=False, tokenizer=None,
              max_doc_size=None, stride=None, **kwargs):
    url = r'https?://(?:www\.)?(?:[-a-zA-Z0-9@:%._\+~#=]{1,256})\.(:?[a-zA-Z0-9()]{1,6})(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)*\b'
    passages = []
    if fields is None:
        num_args = 3
    else:
        num_args = len(fields)
    if 'max_num_documents' in kwargs:
        max_num_documents = kwargs['max_num_documents']
        if max_num_documents is None:
            max_num_documents = 1000000000
    else:
        max_num_documents = 1000000000
    with open(input_file) as in_file:
        if input_file.endswith(".tsv"):
            # We'll assume this is the PrimeQA standard format
            csv_reader = \
                csv.DictReader(in_file, fieldnames=fields, delimiter="\t") \
                    if fields is not None \
                    else csv.DictReader(in_file, delimiter="\t")
            next(csv_reader)
            for ri, row in enumerate(csv_reader):
                if ri >= max_num_documents:
                    break
                assert len(row) in [2, 3, 4], f'Invalid .tsv record (has to contain 2 or 3 fields): {row}'
                if remove_url:
                    row['text'] = re.sub(url, 'URL', row['text'])
                itm = {'text': (row["title"] + ' ' if 'title' in row else '') + row["text"],
                       'id': row['id']}
                if 'title' in row:
                    itm['title'] = row['title']
                if 'relevant' in row:
                    itm['relevant'] = row['relevant']
                if 'answers' in row:
                    itm['answers'] = row['answers'].split("::")
                passages.append(itm)
        elif input_file.endswith('.json'):
            # This should be the SAP json format
            data = json.load(in_file)
            for di, doc in tqdm(enumerate(data), total=min(max_num_documents, len(data)), desc="Reading json documents"):
                doc_id = doc['document_id']
                # doc_title = doc['title']
                if di >= max_num_documents:
                    break
                for passage in doc['passages']:
                    itm = {}
                    title = passage['title']
                    id = f"{doc_id}-{passage['passage_id']}"
                    text = passage['text']
                    if remove_url:
                        text = re.sub(url, 'URL', text)
                    if tokenizer is not None:
                        merged_length = get_tokenized_length(tokenizer=tokenizer, text=text)
                        if merged_length <= max_doc_size:
                            passages.append(
                                {'id': id, 'title': title, 'text': text}
                            )
                        else:
                            # title_len = get_tokenized_length(tokenizer=tokenizer, text=title)
                            maxl = max_doc_size # - title_len
                            psgs = split_passages(text=text, max_length=maxl, stride=stride, tokenizer=tokenizer)
                            for pi, p in enumerate(psgs):
                                passages.append(
                                    {
                                        'id': f"{id}-{pi}",
                                        'title': title,
                                        'text': f"{title}\n{p}"
                                    }
                                )
                    else:
                        passages.append({'id': id, 'title': title, 'text': text})
        elif 'read_sap_qfile' in kwargs or input_file.endswith(".csv"):
            import pandas as pd
            data = pd.read_csv(in_file)
            passages = []
            unmapped_ids = []
            return_unmapped_ids = 'return_unmapped' in kwargs and kwargs['return_unmapped']
            docid_map = kwargs['docid_map'] if 'docid_map' in kwargs else {}
            for i in range(len(data)):
                itm = {}
                itm['id'] = i
                itm['text'] = data.Question[i]
                itm['answers'] = data['Gold answer'][i]
                psgs = []
                ids = []
                for val, loio in [[f'passage {k}', f'loio {k}'] for k in range(1, 4)]:
                    if type(data[val][i]) == str:
                        psgs.append(data[val][i])
                        loio_v = data[loio][i].replace('loio', '')
                        if loio_v in docid_map:
                            if docid_map[loio_v] not in ids:
                                ids.append(docid_map[loio_v])
                        else:
                            ids.append(loio_v)
                            unmapped_ids.append(loio_v)
                itm['passages'] = psgs
                itm['relevant'] = ids
                passages.append(itm)
            if return_unmapped_ids:
                return passages, unmapped_ids
        else:
            raise RuntimeError(f"Unknown file extension: {os.path.splitext(input_file)[1]}")

    return passages


def compute_embedding(model, input_query, normalize_embs):
    query_vector = model.encode(input_query)
    if normalize_embs:
        query_vector = normalize([query_vector])[0]
    return query_vector


class MyEmbeddingFunction:
    def __init__(self, name, batch_size=128):
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda':
            print(f"You are using {device}. This is much slower than using "
                  "a CUDA-enabled GPU. If on Colab you can change this by "
                  "clicking Runtime > Change runtime type > GPU.")
        self.pqa = False
        self.batch_size = batch_size
        if os.path.exists(name):
            raise NotImplemented
            # from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AutoConfig
            # self.queries_to_vectors = None # queries_to_vectors
            # self.model = DPRQuestionEncoder.from_pretrained(
            #     pretrained_model_name_or_path=name,
            #     from_tf = False,
            #     cache_dir=None,)
            # self.model.eval()
            # self.model = self.model.half()
            # self.model.to(device)
            # self.pqa = True
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(name, device=device)
        print('=== done initializing model')

    def get_sentence_embedding_dimension(self):
        return self._model_config.hidden_size

    def __call__(self, texts: Union[List[str], str]) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        return self.encode(texts)

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def encode(self, texts: Union[str, List[str]], batch_size: int = -1) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        if batch_size == -1:
            batch_size = self.batch_size
        if not self.pqa:
            embs = self.model.encode(texts,
                                     show_progress_bar=False \
                                         if isinstance(texts, str) or \
                                            max(len(texts), batch_size) <= 1 \
                                         else True
                                     )
        else:
            raise NotImplemented
            # if batch_size < 0:
            #     batch_size = self.batch_size
            # if len(texts) > batch_size:
            #     embs = []
            #     for i in tqdm(range(0, len(texts), batch_size)):
            #         i_end = min(i + batch_size, len(texts))
            #         tems = self.queries_to_vectors(self.tokenizer,
            #                                        self.model,
            #                                        texts[i:i_end],
            #                                        max_query_length=500).tolist()
            #         embs.extend(tems)
            # else:
            #     embs = self.queries_to_vectors(self.tokenizer, self.model, texts, max_query_length=500).tolist()
        return embs


def normalize(passage_vectors):
    return [v / np.linalg.norm(v) for v in passage_vectors if np.linalg.norm(v) > 0]


def compute_score(input_queries, results):
    from rouge_score.rouge_scorer import RougeScorer
    scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    if "relevant" not in input_queries[0] or input_queries[0]['relevant'] is None:
        print("The input question file does not contain answers. Please fix that and restart.")
        sys.exit(12)
    ranks = [int(r) for r in args.ranks.split(",")]
    scores = {r: 0 for r in ranks}
    pscores = {r: 0 for r in ranks}  # passage scores
    gt = {-1: -1}
    for q in input_queries:
        gt[q['id']] = {id: 1 for id in q['relevant']}

    def skip(out_ranks, record, rid):
        qid = record[0]
        while rid < len(out_ranks) and out_ranks[rid][0] == qid:
            rid += 1
        return rid

    def reverse_map(input_queries):
        rq_map = {}
        for i, q in enumerate(input_queries):
            rq_map[q['id']] = i
        return rq_map

    def update_scores(ranks, rnk, val, op, scores):
        j = 0
        while j < len(ranks) and ranks[j] < rnk:
            j += 1
        for k in ranks[j:]:
            # scores[k] += 1
            scores[k] = op([scores[k], val])

    def get_doc_id(label):
        index = label.find("-")
        if index >= 0:
            return label[:index]
        else:
            return label

    tmp_scores = scores.copy()
    tmp_pscores = pscores.copy()
    prev_id = -1

    num_eval_questions = 0
    for rid, record in tqdm(enumerate(results),
                            total=len(results),
                            desc='Evaluating questions: '):
        qid = record['qid']
        query = input_queries[qid]
        if '-1' in gt[qid]:
            continue
        num_eval_questions += 1
        tmp_scores = {r: 0 for r in ranks}
        tmp_pscores = {r: 0 for r in ranks}
        for aid, answer in enumerate(record['answers']):
            docid = get_doc_id(answer['id'])

            if str(docid) in gt[qid]:  # Great, we found a match.
                update_scores(ranks, aid, 1, sum, tmp_scores)
            scr = max(
                [
                    scorer.score(passage, answer['text'])['rouge1'].fmeasure for passage in query['passages']
                ]
            )
            update_scores(ranks, aid, scr, max, tmp_pscores)

        for r in ranks:
            scores[r] += int(tmp_scores[r] >= 1)
            pscores[r] += tmp_pscores[r]

    res = {"num_ranked_queries": num_eval_questions,
           "num_judged_queries": num_eval_questions,
           "doc_scores":
               {r: int(1000 * scores[r] / num_eval_questions) / 1000.0 for r in ranks},
           "passage_scores":
               {r: int(1000 * pscores[r] / num_eval_questions) / 1000.0 for r in ranks}
           }
    return res


def check_index_rebuild():
    while True:
        r = input("Are you sure you want to recreate the index? It might take a long time!! Say 'yes' or 'no':").strip()
        if r == 'no':
            print("OK - exiting. Run with '--actions r'")
            sys.exit(0)
        elif r == 'yes':
            break
        else:
            print(f"Please type 'yes' or 'no', not {r}!")


if __name__ == '__main__':
    parser = ArgumentParser(description="Script to create/use ElasticSearch indices")
    parser.add_argument('--input_passages', '-p', default=None)
    parser.add_argument('--input_queries', '-q', default=None)

    parser.add_argument('--db_engine', '-e', default='es-dense',
                        choices=['es-dense', 'es-elser'], required=False)
    parser.add_argument('--output_file', '-o', default=None, help="The output rank file.")

    parser.add_argument('--top_k', '-k', type=int, default=10, )
    parser.add_argument('--model_name', '-m', default='all-MiniLM-L6-v2')
    parser.add_argument('--actions', default="ir",
                        help="The actions that can be done: i(ingest), r(retrieve), R(rerank)")
    parser.add_argument("--normalize_embs", action="store_true", help="If present, the embeddings are normalized.")
    parser.add_argument("--evaluate", action="store_true",
                        help="If present, evaluates the results based on test data, at the provided ranks.")
    parser.add_argument("--ranks", default="1,5,10,100", help="Defines the R@i evaluation ranks.")
    parser.add_argument('--data', default=None, type=str, help="The directory containing the data to use. The passage "
                                                               "file is assumed to be args.data/psgs.tsv and "
                                                               "the question file is args.data/questions.tsv.")
    parser.add_argument("--ingestion_batch_size", default=40, type=int,
                        help="For elastic search only, sets the ingestion batch "
                             "size (default 40).")
    parser.add_argument("--replace_links", action="store_true", default=False,
                        help="If turned on, it will replace urls in text with URL<no>")
    parser.add_argument("--max_doc_length", type=int, default=None,
                        help="If provided, the documents will be split into chunks of <max_doc_length> "
                             "*word pieces* (in regular English text, about 2 word pieces for every word). "
                             "If not provided, the passages in the file will be ingested, truncated.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Argument that works in conjunction with --max_doc_length: it will define the "
                             "increment of the window start while tiling the documents.")
    parser.add_argument("--max_num_documents", type=int, default=None,
                        help="If defined, it will restrict the ingestion to the first <max_num_documents> documents")
    parser.add_argument("--docid_map", type=str, default=None,
                        help="If defined, this provides a link to a file mapping docid values to loio values.")
    parser.add_argument("-I", "--index_name", type=str, default=None,
                        help="Defines the index name to use. If not specified, it is built as " \
                             "{args.data}_{args.db_engine}_{args.model_name if args.db_engine=='es-dense' else 'elser'}_index")

    args = parser.parse_args()
    if args.index_name is None:
        index_name = (
            f"{args.data}_{args.db_engine}_{args.model_name if args.db_engine == 'es-dense' else 'elser'}_index").lower()
    else:
        index_name = args.index_name.lower()

    index_name = re.sub('[^a-z0-9]', '-', index_name)

    do_ingest = 'i' in args.actions
    do_retrieve = 'r' in args.actions
    do_rerank = 'R' in args.actions

    model = None
    if args.db_engine == "es-dense" or args.max_doc_length is not None:
        import torch

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda':
            print(f"You are using {device}. This is much slower than using "
                  "a CUDA-enabled GPU. If on Colab you can change this by "
                  "clicking Runtime > Change runtime type > GPU.")

        batch_size = 64
        model = MyEmbeddingFunction(args.model_name)

    ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
    if ELASTIC_PASSWORD is None or ELASTIC_PASSWORD == "":
        print(f"You need to define the environment variable ELASTIC_PASSWORD for the elastic user! Define it and restart.")
        sys.exit(11)
    client = Elasticsearch(
        cloud_id="sap-deployment:dXMtZWFzdC0xLmF3cy5mb3VuZC5pbzo0NDMkOGYwZTRiNTBmZGI1NGNiZGJhYTk3NjhkY2U4N2NjZTAkODViMzExOTNhYTQwNDgyN2FhNGE0MmRiYzg5ZDc4ZjE=",
        basic_auth=("elastic", ELASTIC_PASSWORD)
        )

    if do_ingest:
        max_documents = args.max_num_documents

        input_passages = read_data(args.input_passages,
                                   fields=["id", "text", "title"],
                                   remove_url=args.replace_links,
                                   max_doc_size=args.max_doc_length,
                                   stride=args.stride,
                                   tokenizer=model.tokenizer if model is not None else None,
                                   max_num_documents=max_documents,
                                   )
        if max_documents is not None and max_documents > 0:
            input_passages = input_passages[:max_documents]

        hidden_dim = -1
        if args.db_engine == "es-dense":
            passage_vectors = model.encode([passage['text'] for passage in input_passages], batch_size=batch_size)

            hidden_dim = len(passage_vectors[0])
            if args.normalize_embs:
                passage_vectors = normalize(passage_vectors)

        logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)

        if args.db_engine == "es-dense":
            mappings = {
                "properties": {
                    "title": {"type": "text", "analyzer": "english"},
                    "text": {"type": "text", "analyzer": "english"},
                    "vector": {"type": "dense_vector", "dims": hidden_dim,
                               "similarity": "cosine", "index": "true"},
                }
            }
            if client.indices.exists(index=index_name):
                check_index_rebuild()
                client.options(ignore_status=[400, 404]).indices.delete(index=index_name)
            client.indices.create(index=index_name, mappings=mappings)
            logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
            bulk_batch = args.ingestion_batch_size

            num_passages = len(input_passages)
            t = tqdm(total=num_passages, desc="Ingesting dense documents: ", smoothing=0.05)
            for k in range(0, num_passages, bulk_batch):
                actions = [{"_index": index_name,
                         "_id": row['id'],
                         "_source": {
                             'text': row['text'],
                             'title': row['title'],
                             'vector': passage_vectors[pi+k]
                         }}
                        for pi, row in enumerate(input_passages[k:min(k+bulk_batch, num_passages)])
                        ]
                try:
                    bulk(client, actions=actions)
                except Exception as e:
                    print(f"Got an error in indexing: {e}")
                t.update(bulk_batch)
            t.close()

            # for ri, row in tqdm(enumerate(input_passages), desc="Indexing es-dense", total=len(input_passages)):
                # doc = {'text': row['text'],
                #        'title': row['title'],
                #        'vector': passage_vectors[ri]}
                # client.index(index=index_name, id=row['id'], document=doc)
        elif args.db_engine == "es-elser":
            mappings = {
                "properties": {
                    "ml.tokens": {
                        "type": "rank_features"
                    },
                    "title": {"type": "text", "analyzer": "english"},
                    "text": {"type": "text", "analyzer": "english"},
                }
            }
            processors = [
                {
                    "inference": {
                        "model_id": ".elser_model_1",
                        "target_field": "ml",
                        "field_map": {
                            "text": "text_field"
                        },
                        "inference_config": {
                            "text_expansion": {
                                "results_field": "tokens"
                            }
                        }
                    }}
            ]
            bulk_batch = args.ingestion_batch_size
            if client.indices.exists(index=index_name):
                check_index_rebuild()
                client.options(ignore_status=[400, 404]).indices.delete(index=index_name)
            client.indices.create(index=f"{index_name}", mappings=mappings)
            client.ingest.put_pipeline(processors=processors, id='elser-v1-test')
            actions = []
            for ri, row in tqdm(enumerate(input_passages), total=len(input_passages), desc="Indexing passages"):
                actions.append({
                    "_index": index_name,
                    "_id": row['id'],
                    "_source": {
                        'text': row['text'],
                        'title': row['title']
                    }
                }
                )
                if ri % bulk_batch == bulk_batch - 1:
                    failures = 0
                    while failures < 5:
                        try:
                            res = bulk(client=client, actions=actions, pipeline="elser-v1-test")
                            break
                        except Exception as e:
                            print(f"Got an error in indexing: {e}, {len(actions)} {res}")
                        failures += 5
                    actions = []
            if len(actions) > 0:
                try:
                    bulk(client=client, actions=actions, pipeline="elser-v1-test")
                except Exception as e:
                    print(f"Got an error in indexing: {e}, {len(actions)}")

    ### QUERY TIME

    if do_retrieve:
        loio2docid = {}
        if args.docid_map is not None:
            with open(args.docid_map) as inp:
                for line in inp:
                    a = line.split()
                    loio2docid[a[1]] = a[0]

        if args.evaluate:
            input_queries = read_data(args.input_queries, fields=["id", "text", "relevant", "answers"],
                                      docid_map=loio2docid)
        else:
            input_queries = read_data(args.input_queries, fields=["id", "text"])

        result = []
        if args.db_engine == "es-dense":
            for query_number in tqdm(range(len(input_queries))):
                query_vector = compute_embedding(model, input_queries[query_number]['text'], args.normalize_embs)
                qid = input_queries[query_number]['id']
                query = {
                    "field": "vector",
                    "query_vector": query_vector,
                    "k": args.top_k,
                    "num_candidates": 1000,
                }
                res = client.search(
                    index=index_name,
                    knn=query,
                    source_excludes=['vector']
                )
                rout = []
                for rank, r in enumerate(res._body['hits']['hits']):
                    rout.append({'id': r['_id'], 'score': r['_score'], 'text': r['_source']['text']})
                result.append({'qid': qid, 'text': input_queries[query_number]['text'], "answers": rout})
        elif args.db_engine == "es-elser":
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
                res = client.search(
                    index=index_name,
                    query=query,
                    size=args.top_k,
                )
                rout = []
                for rank, r in enumerate(res._body['hits']['hits']):
                    rout.append({'id': r['_id'], 'score': r['_score'], 'text': r['_source']['text']})
                result.append({'qid': qid, 'text': input_queries[query_number]['text'], "answers": rout})

        if do_rerank:
            pass

        if args.output_file is not None:
            with open(args.output_file, 'w') as out:
                json.dump(result, out, indent=2)

        if args.evaluate:
            score = compute_score(input_queries, result)

            with open(args.output_file.replace(".json", ".metrics"), "w") as out:
                json.dump(score, out, indent=2)
