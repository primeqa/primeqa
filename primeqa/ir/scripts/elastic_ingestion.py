import os, re, json, csv
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
from typing import List, Union, Any
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import logging
import sys
import pyizumo
import unicodedata
import copy

nlp = None
product_counts = {}
import urllib3

urllib3.disable_warnings()

languages = ['en', 'es', 'fr', 'pt', 'ja', 'de', 'ja-new']

settings = {}
coga_mappings = {}
docname2url = {}
normalize_text = False


def setup_argparse():
    parser = ArgumentParser(description="Script to create/use ElasticSearch indices")
    parser.add_argument('--input_passages', '-p', nargs="+", default=None)
    parser.add_argument('--input_queries', '-q', default=None)

    parser.add_argument('--db_engine', '-e', default='es-dense',
                        choices=['es-dense', 'es-elser', 'es-bm25', 'es-dense'], required=False)
    parser.add_argument('--output_file', '-o', default=None, help="The output rank file.")
    parser.add_argument("--model_on_server", action="store_true",
                        help="If present, the given model is assumed to exist on the ES server.")
    parser.add_argument("--hybrid", choices=['rrf'], default="rrf",
                        help="The type of hybrid combination")
    parser.add_argument('--top_k', '-k', type=int, default=10, )
    parser.add_argument('--model_name', '-m')
    parser.add_argument('--actions', default="ir",
                        help="The actions that can be done: i(ingest), r(retrieve), R(rerank), u(update)")
    parser.add_argument("--normalize_embs", action="store_true", help="If present, the embeddings are normalized.")
    parser.add_argument("--evaluate", action="store_true",
                        help="If present, evaluates the results based on test data, at the provided ranks.")
    parser.add_argument("--ranks", default="1,5,10,100", help="Defines the R@i evaluation ranks.")
    parser.add_argument('--data', default=None, type=str, help="The directory containing the data to use. The passage "
                                                               "file is assumed to be args.data/psgs.tsv and "
                                                               "the question file is args.data/questions.tsv.")
    parser.add_argument("--data_type", default="auto", type=str, choices=["auto", 'pqa', 'sap', 'beir', 'rh'],
                        help=("The type of the dataset to use. If auto, then the type will be determined"
                              "by the file extension: .tsv->pqa, .json|.jsonl -> sap, csv -> SAP question"))
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
    parser.add_argument("--doc_based", action="store_true", default=False,
                        help="If present, the document text will be ingested, otherwise the ingestion will be done"
                             " at passage level.")
    parser.add_argument("--hana_file2url", type=str, default=None,
                        help="The file mapping the docid to the url to the title")
    parser.add_argument("--remove_stopwords", action="store_true", default=False,
                        help="If defined, the stopwords are removed from text before indexing.")
    parser.add_argument("--docids_to_ingest", default=None, help="If provided, only the documents with the "
                                                                 "ids in the file will be added.")
    parser.add_argument("--product_name", default=None, help="If set, this product name will be used "
                                                             "for all documents")
    parser.add_argument("--server", default="CONVAI",
                        choices=['SAP', 'CONVAI', 'SAP_TEST', 'AILANG', 'local', 'RESCONVAI'],
                        help="The server to connect to.")
    parser.add_argument("--lang", "--language_code", default="en", choices=languages)
    parser.add_argument("--host", default=None, help="Gives the IP for the ES server; can be used to "
                                                     "override the defaults based on --server")
    parser.add_argument("-N", "--normalize_text", default=False, help="If provided, the text is normalized "
                                                                      "for UTF8 encoding.")

    return parser


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


def get_pyizumo_tokenized_text(text=None, tokenized_text=None, language_code=None):
    tok_text = []

    if not tokenized_text:
        if text == None or language_code == None:
            raise RuntimeError("text and language_code needed if tokenized_text not provided")
        global nlp
        if not nlp:
            nlp = pyizumo.load(language_code, parsers=['token', 'sentence'])
        tokenized_text = nlp(text)

    for sent in tokenized_text.sentences:
        sent_tokens = []
        for tok in sent.tokens:
            if 'components' in tok.properties:
                for part in tok.properties['components']:
                    sent_tokens.append(part['text'])
            else:
                sent_tokens.append(tok.text)
        tok_text.append(" ".join(sent_tokens))

    token_text = "\n".join(tok_text)

    return token_text


def split_text(text: str, tokenizer, title: str = "", max_length: int = 512, stride: int = None, language_code='en') \
        -> tuple[list[str], list[list[int | Any]]]:
    """
    Method to split a text into pieces that are of a specified <max_length> length, with the
    <stride> overlap, using a HF tokenizer.
    :param text: str - the text to split
    :param tokenizer: HF Tokenizer
       - the tokenizer to do the work of splitting the words into word pieces
    :param title: str - the title of the document
    :param max_length: int - the maximum length of the resulting sequence
    :param stride: int - the overlap between windows
    """
    global nlp
    text = re.sub(r' {2,}', ' ', text, flags=re.MULTILINE)  # remove multiple spaces.
    if max_length is not None:
        # res = tokenizer(text, max_length=max_length, stride=stride,
        #                 return_overflowing_tokens=True, truncation=True)
        tok_len = get_tokenized_length(tokenizer, text)
        if tok_len <= max_length:
            return [text], [[0, len(text)]]
        else:
            if title:  # make space for the title in each split text.
                ltitle = get_tokenized_length(tokenizer, title)
                max_length -= ltitle
                ind = text.find(title)
                if ind == 0:
                    text = text[ind + len(title):]

            if not nlp:
                nlp = pyizumo.load(language_code, parsers=['token', 'sentence'])
            parsed_text = nlp(text)

            tsizes = []
            begins = []
            ends = []
            tokens = []
            for sent in parsed_text.sentences:
                stext = sent.text
                slen = get_tokenized_length(tokenizer, stext)
                if slen > max_length:
                    too_long = [[t for t in sent.tokens]]
                    too_long[0].reverse()
                    while len(too_long) > 0:
                        tl = too_long.pop()
                        ll = get_tokenized_length(tokenizer, text[tl[-1].begin:tl[0].end])
                        if ll <= max_length:
                            tsizes.append(ll)
                            begins.append(tl[-1].begin)
                            ends.append(tl[0].end)
                        else:
                            if len(tl) > 1:  # Ignore really long words
                                mid = int(len(tl) / 2)
                                too_long.extend([tl[:mid], tl[mid:]])
                            else:
                                pass
                else:
                    tsizes.append(slen)
                    begins.append(sent.begin)
                    ends.append(sent.end)

            intervals = compute_intervals(tsizes, max_length, stride)

            positions = [[begins[p[0]], ends[p[1]]] for p in intervals]
            texts = [text[p[0]:p[1]] for p in positions]
            return texts, positions


MAX_TRIED = 10000


def compute_intervals(segment_lengths: List[int], max_length: int, stride: int) -> List[List[int | Any]]:
    def check_for_cycling(segments: List[List[int | Any]], prev_start_index: int) -> None:
        if len(segments) > 0 and segments[-1][0] == prev_start_index:
            raise RuntimeError(f"You have a problem with the splitting - it's cycling!: {segments[-3:]}")

    def check_excessive_tries(number_of_tries: int) -> bool:
        if number_of_tries > MAX_TRIED:
            print(f"Too many tried - probably something is wrong with the document.")
            return True
        else:
            return False

    current_index = 1
    current_total_length = segment_lengths[0]
    prev_start_index = 0
    segments = []
    number_of_tries = 0

    while current_index < len(segment_lengths):
        if current_total_length + segment_lengths[current_index] > max_length:
            check_for_cycling(segments, prev_start_index)
            if check_excessive_tries(number_of_tries):
                return segments
            if segments is None:
                break
            number_of_tries += 1
            segments.append([prev_start_index, current_index - 1])

            if current_index > 1 and segment_lengths[current_index - 1] + segment_lengths[current_index] <= max_length:
                overlap_index = current_index - 1
                overlap = 0
                max_length_tmp = max_length - segment_lengths[current_index]
                while overlap_index > 0:
                    overlap += segment_lengths[overlap_index]
                    if overlap < stride and overlap + segment_lengths[overlap_index - 1] <= max_length_tmp:
                        overlap_index -= 1
                    else:
                        break
                current_index = overlap_index
            prev_start_index = current_index
            current_total_length = 0
        else:
            current_total_length += segment_lengths[current_index]
            current_index += 1
    segments.append([prev_start_index, len(segment_lengths) - 1])

    return segments


def get_tokenized_length(tokenizer, text):
    """
    Returns the size of the <text> after being tokenized by <tokenizer>
    :param tokenizer: Tokenizer - the tokenizer to convert text to word pieces
    :param text: str - the input text
    :return the length (in word pieces) of the tokenized text.
    """
    if tokenizer is not None:
        toks = tokenizer(text)
        return len(toks['input_ids'])
    else:
        return -1


def process_product_id(url_fields, uniform_product_name, data_type):
    """
    Process the product ID based on the given fields, uniform product name, and data type.

    Parameters:
    - fields (list): A list of fields.
    - uniform_product_name (str): The uniform product name.
    - data_type (str): The data type.

    Returns:
    - str: The processed product ID.

    Example Usage:
    ```python
    fields = ["field1", "field2", "field3"]
    uniform_product_name = "Uniform Product"
    data_type = "sap"

    result = process_product_id(fields, uniform_product_name, data_type)
    print(result)  # Output: ""

    data_type = "other"

    result = process_product_id(fields, uniform_product_name, data_type)
    print(result)  # Output: ""
    ```
    """
    if data_type == "sap":
        productId = "" if len(url_fields) == 0 \
            else url_fields[-3] if (len(url_fields) > 3 and url_fields[-3] != '#') \
            else 'SAP_BUSINESS_ONE'
        if uniform_product_name:
            productId = uniform_product_name
        if productId.startswith("SAP_SUCCESSFACTORS"):
            productId = "SAP_SUCCESSFACTORS"
        return productId
    else:
        return ""


def process_url(doc_url, data_type):
    if data_type == "sap":
        url = re.sub(r'\?locale=.*', "", doc_url)
        fields = url.split("/")
        fields[-1] = fields[-1].replace(".html", "")
        return url, fields
    else:
        return "", ["", "", "", "", "", ""]


def update_product_counts(product_counts, productId):
    if productId not in product_counts:
        product_counts[productId] = 1
    else:
        product_counts[productId] += 1


def process_text(id, title, text, max_doc_size, stride, remove_url=True,
                 tokenizer=None,
                 doc_url=None,
                 uniform_product_name=None,
                 data_type="sap"
                 ):
    """
    Convert a given document or passage (from 'output.json') to a dictionary, splitting the text as necessary.
    :param id: str - the prefix of the id of the resulting piece/pieces
    :param title: str - the title of the new piece
    :param text: the input text to be split
    :param max_doc_size: int - the maximum size (in word pieces) of the resulting sub-document/sub-passage texts
    :param stride: int - the stride/overlap for consecutive pieces
    :param remove_url: Boolean - if true, URL in the input text will be replaced with "URL"
    :param tokenizer: Tokenizer - the tokenizer to use while splitting the text into pieces
    :param doc_url: str - the url of the document.
    :param uniform_product_name: str - if not None, all documents will receive this productId
    :return - a list of indexable items, each containing a title, id, text, and url.
    """
    global product_counts, normalize_text
    pieces = []
    # Refactoring: extracted url processing to a separate function
    full_url, fields = process_url(doc_url, data_type)

    # Refactoring: extracted productId processing to a separate function
    productId = process_product_id(fields, uniform_product_name, data_type)

    # Refactoring: extracted product_counts updating to a separate function
    update_product_counts(product_counts, productId)
    itm = {
        'productId': productId,
        'deliverableLoio': ("" if doc_url == "" else fields[-2]),
        'filePath': "" if doc_url == "" else fields[-1],
        'title': title,
        'url': doc_url,
        'app_name': "",
    }
    #
    url = r'https?://(?:www\.)?(?:[-a-zA-Z0-9@:%._\+~#=]{1,256})\.(:?[a-zA-Z0-9()]{1,6})(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)*\b'
    if text.find("With this app") >= 0 or text.find("App ID") >= 0:
        itm['app_name'] = title
    if not text.startswith(title):
        expanded_text = f"{title}\n{text}"
    else:
        expanded_text = text

    if remove_url:
        # The normalization below deals with some issue in the re library - it would get stuck
        # if the URL has some strange chars, like `\xa0`.
        if normalize_text:
            text = re.sub(url, 'URL', unicodedata.normalize("NFKC", text))
        else:
            text = re.sub(url, 'URL', text)

    if tokenizer is not None:
        merged_length = get_tokenized_length(tokenizer=tokenizer, text=expanded_text)
        if merged_length <= max_doc_size:
            itm.update({'id': f"{id}-0-{len(text)}", 'text': expanded_text})
            pieces.append(itm.copy())
        else:
            maxl = max_doc_size  # - title_len
            psgs, inds = split_text(text=text, max_length=maxl, title=title,
                                    stride=stride, tokenizer=tokenizer)
            for pi, (p, index) in enumerate(zip(psgs, inds)):
                itm.update({
                    'id': f"{id}-{index[0]}-{index[1]}",
                    'text': f"{title}\n{p}"
                })
                pieces.append(itm.copy())
    else:
        itm.update({'id': id, 'text': expanded_text})
        pieces.append(itm.copy())
    return pieces


def get_attr(_args, val, default=None):
    if val in _args and _args[val] is not None:
        return _args[val]
    else:
        return default


def remove_stopwords(text: str, lang, do_replace: bool = False) -> str:
    global stopwords, settings
    if not do_replace:
        return text
    else:
        if stopwords is None:
            stopwords = re.compile(
                "\\b(?:" + "|".join(settings["analysis"]["filter"][f"{lang}_stop"]["stopwords"]) + ")\\b",
                re.IGNORECASE)
        return re.sub(r' {2,}', ' ', re.sub(stopwords, " ", text))


def read_data(input_files, lang, fields=None, remove_url=False, tokenizer=None,
              max_doc_size=None, stride=None, **kwargs):
    passages = []
    doc_based = get_attr(kwargs, 'doc_based')
    docid_map = get_attr(kwargs, 'docid_map', default={})
    max_num_documents = get_attr(kwargs, 'max_num_documents', default=1000000000)
    url = r'https?://(?:www\.)?(?:[-a-zA-Z0-9@:%._\+~#=]{1,256})\.(:?[a-zA-Z0-9()]{1,6})(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)*\b'
    data_type = get_attr(kwargs, 'data_type')
    if fields is None:
        num_args = 3
    else:
        num_args = len(fields)
    if isinstance(input_files, list):
        files = input_files
    elif isinstance(input_files, str):
        files = [input_files]
    else:
        raise RuntimeError(f"Unsupported type for {input_files}")
    docname2url = get_attr(kwargs, 'docname2url', None)
    docs_read = 0
    remv_stopwords = get_attr(kwargs, 'remove_stopwords', False)
    unmapped_ids = []
    return_unmapped_ids = get_attr(kwargs, 'return_unmapped')

    for input_file in files:
        docs_read = 0
        print(f"Reading {input_file}")
        with open(input_file) as in_file:
            if input_file.endswith(".tsv"):
                # We'll assume this is the PrimeQA standard format
                csv_reader = \
                    csv.DictReader(in_file, fieldnames=fields, delimiter="\t") \
                        if fields is not None \
                        else csv.DictReader(in_file, delimiter="\t")
                next(csv_reader)
                for ri, row in tqdm(enumerate(csv_reader)):
                    if ri >= max_num_documents:
                        break
                    assert len(row) in [2, 3, 4], f'Invalid .tsv record (has to contain 2 or 3 fields): {row}'
                    if 'answers' in row:
                        if remove_url:
                            row['text'] = remove_stopwords(re.sub(url, lang, 'URL', row['text']), remv_stopwords)
                        itm = {'text': (row["title"] + ' ' if 'title' in row else '') + row["text"],
                               'id': row['id']}
                        if 'title' in row:
                            itm['title'] = remove_stopwords(row['title'], lang, remv_stopwords)
                        if 'relevant' in row:
                            itm['relevant'] = row['relevant'].split(",")
                        if 'answers' in row:
                            itm['answers'] = row['answers'].split("::")
                            itm['passages'] = itm['answers']
                        passages.append(itm)
                    else:
                        passages.extend(
                            process_text(id=row['id'],
                                         title=remove_stopwords(fix_title(row['title']), lang,
                                                                remv_stopwords) if 'title' in row else '',
                                         text=remove_stopwords(row['text'], lang, remv_stopwords),
                                         max_doc_size=max_doc_size,
                                         stride=stride,
                                         remove_url=remove_url,
                                         tokenizer=tokenizer,
                                         doc_url=url,
                                         uniform_product_name=None,
                                         data_type=data_type
                                         ))
            elif input_file.endswith('.json') or input_file.endswith(".jsonl"):
                # This should be the SAP or BEIR json format
                if input_file.endswith('.json'):
                    data = json.load(in_file)
                else:
                    data = [json.loads(line) for line in open(input_file).readlines()]
                uniform_product_name = get_attr(kwargs, 'uniform_product_name')
                docid_filter = get_attr(kwargs, 'docid_filter', [])
                # data_type = get_attr(kwargs, 'data_type', 'sap')
                if data_type in ['auto', 'sap']:
                    txtname = "document"
                    docidname = "document_id"
                    titlename = "title"
                    data_type = "sap"
                elif data_type == "beir":
                    txtname = "text"
                    docidname = "_id"
                    titlename = 'title'

                for di, doc in tqdm(enumerate(data),
                                    total=min(max_num_documents, len(data)),
                                    desc="Reading json documents",
                                    smoothing=0.05):
                    if di >= max_num_documents:
                        break
                    docid = doc[docidname]

                    if ".txt" in docid:
                        docid = docid.replace(".txt", "")

                    if docid_filter != [] and docid not in docid_filter:
                        continue
                    url = doc['document_url'] if 'document_url' in doc else \
                        doc['url'] if 'url' in doc else ""
                    title = doc[titlename] if 'title' in doc else None
                    if title is None:
                        title = ""
                    if docname2url and docid in docname2url:
                        url = docname2url[docid]
                        title = docname2title[docid]

                    try:
                        if doc_based:
                            passages.extend(
                                process_text(id=doc[docidname],
                                             title=remove_stopwords(fix_title(title), lang, remv_stopwords),
                                             text=remove_stopwords(doc[txtname], remv_stopwords),
                                             max_doc_size=max_doc_size,
                                             stride=stride,
                                             remove_url=remove_url,
                                             tokenizer=tokenizer,
                                             doc_url=url,
                                             uniform_product_name=uniform_product_name,
                                             data_type=data_type
                                             ))
                        else:
                            for passage in doc['passages']:
                                passages.extend(
                                    process_text(id=f"{doc[docidname]}-{passage['passage_id']}",
                                                 title=remove_stopwords(passage[titlename], remv_stopwords),
                                                 text=remove_stopwords(passage[txtname], remv_stopwords),
                                                 max_doc_size=max_doc_size,
                                                 stride=stride,
                                                 remove_url=remove_url,
                                                 tokenizer=tokenizer,
                                                 doc_url=url,
                                                 uniform_product_name=uniform_product_name,
                                                 data_type=data_type
                                                 ))
                    except Exception as e:
                        print(f"Error at line {di}: {e}")
                        raise e
                    docs_read += 1
            elif get_attr(kwargs, 'read_sap_qfile', default=False) or input_file.endswith(".csv"):
                import pandas as pd
                data = pd.read_csv(in_file)
                passages = []
                docid_map = get_attr(kwargs, 'docid_map', default={})
                for i in range(len(data)):
                    itm = {}
                    itm['id'] = i
                    itm['text'] = remove_stopwords(data.Question[i].strip(), remv_stopwords)
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
        max_num_documents -= docs_read

    if return_unmapped_ids:
        return passages, unmapped_ids
    else:
        return passages


def fix_title(title):
    return re.sub(r' {2,}', ' ', title.replace(" | SAP Help Portal", ""))


def compute_embedding(model, input_query, normalize_embs):
    query_vector = model.encode(input_query)
    if normalize_embs:
        query_vector = normalize([query_vector])[0]
    return query_vector


class MyEmbeddingFunction:
    def __init__(self, name, batch_size=128):
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # "mps" if torch.backends.mps.is_available() else 'cpu'
        if device == 'cpu':
            print(f"You are using {device}. This is much slower than using "
                  "a CUDA-enabled GPU. If on Colab you can change this by "
                  "clicking Runtime > Change runtime type > GPU.")
        self.pqa = False
        self.batch_size = batch_size
        # if os.path.exists(name):
        #     raise NotImplemented
        #     # from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AutoConfig
        #     # self.queries_to_vectors = None # queries_to_vectors
        #     # self.model = DPRQuestionEncoder.from_pretrained(
        #     #     pretrained_model_name_or_path=name,
        #     #     from_tf = False,
        #     #     cache_dir=None,)
        #     # self.model.eval()
        #     # self.model = self.model.half()
        #     # self.model.to(device)
        #     # self.pqa = True
        # else:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device=device)
        print('=== done initializing model')

    def __call__(self, texts: Union[List[str], str]) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        return self.encode(texts)

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def encode(self, texts: Union[str, List[str]], _batch_size: int = -1) -> \
            Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        embs = []
        if _batch_size == -1:
            _batch_size = self.batch_size
        if not self.pqa:
            embs = self.model.encode(texts,
                                     show_progress_bar=False \
                                         if isinstance(texts, str) or \
                                            max(len(texts), _batch_size) <= 1 \
                                         else True,
                                     normalize_embeddings=True
                                     ).tolist()
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
        if isinstance(q['relevant'], list):
            gt[q['id']] = {id: 1 for id in q['relevant']}
        else:
            gt[q['id']] = {q['relevant']: 1}

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
        while j < len(ranks) and ranks[j] <= rnk:
            j += 1
        for k in ranks[j:]:
            # scores[k] += 1
            scores[k] = op([scores[k], val])

    def get_doc_id(label):
        # find - from right side because id may have -
        index = label.rfind("-", 0, label.rfind("-"))
        if index >= 0:
            return label[:index]
        else:
            return label

    rqmap = reverse_map(input_queries)

    num_eval_questions = 0
    for rid, record in tqdm(enumerate(results),
                            total=len(results),
                            desc='Evaluating questions: '):
        qid = record['qid']
        query = input_queries[rqmap[qid]]
        if '-1' in gt[qid]:
            continue
        num_eval_questions += 1
        tmp_scores = {r: 0 for r in ranks}
        tmp_pscores = {r: 0 for r in ranks}
        for aid, answer in enumerate(record['answers']):
            if aid >= ranks[-1]:
                break
            docid = get_doc_id(answer['id'])

            if str(docid) in gt[qid]:  # Great, we found a match.
                update_scores(ranks, aid, 1, sum, tmp_scores)
            if len(query['passages']) == 0:
                scr = 0.
            else:
                scr = max(
                    [
                        scorer.score(passage, answer['text'])['rouge1'].recall for passage in query['passages']
                    ]
                )
            update_scores(ranks, aid, scr, max, tmp_pscores)

        for r in ranks:
            scores[r] += int(tmp_scores[r] >= 1)
            pscores[r] += tmp_pscores[r]

    _result = {"num_ranked_queries": num_eval_questions,
               "num_judged_queries": num_eval_questions,
               "doc_scores":
                   {r: int(1000 * scores[r] / num_eval_questions) / 1000.0 for r in ranks},
               "passage_scores":
                   {r: int(1000 * pscores[r] / num_eval_questions) / 1000.0 for r in ranks}
               }
    return _result


def check_index_rebuild(index_name):
    while True:
        r = input(
            f"Are you sure you want to recreate the index {index_name}? It might take a long time!! Say 'yes' or 'no':").strip()
        if r == 'no':
            print("OK - exiting. Run with '--actions r'")
            sys.exit(0)
        elif r == 'yes':
            break
        else:
            print(f"Please type 'yes' or 'no', not {r}!")


def create_update_index(client, index_name, settings, mappings, do_update=False):
    if client.indices.exists(index=index_name):
        if not do_update:
            check_index_rebuild(index_name)
            client.options(ignore_status=[400, 404]).indices.delete(index=index_name)
        else:
            print(f"Using existent index {index_name}.")
    else:
        if do_update:
            print("You are trying to update an index that does not exist "
                  "- will ignore your command and create the index.")
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, mappings=mappings, settings=settings)


def init_settings():
    def union(a, b):
        if a == {}:
            return copy.deepcopy(b)
        else:
            c = copy.deepcopy(a)
            for k in b.keys():
                if k in a:
                    c[k] = union(a[k], b[k])
                else:
                    c[k] = copy.deepcopy(b[k])
        return c

    global settings, coga_mappings, standard_mappings
    config = json.load(open(f"{os.path.join(os.path.dirname(__file__), 'elastic_config.json')}"))
    standard_mappings = config['settings']['standard']
    for lang in languages:
        settings[lang] = union(config['settings']['common'],
                               config['settings'][lang if lang in config['settings'] else 'en']
                               )
        coga_mappings[lang] = union(config['mappings']['common'],
                                    config['mappings'][lang if lang in config['mappings'] else 'en']
                                    )


def init_settings_lang(lang):
    global settings, coga_mappings

    lang_stemmer = {
        'es': "light_spanish",
        'fr': "light_french",
        'en': "light_english",
        'de': "light_german",
        'pt': "light_portuguese",
    }

    lang_stop = {
        'es': "_spanish_",
        'fr': "_french_",
        'en': ["a", "about", "all", "also", "am", "an", "and", "any", "are", "as", "at",
               "be", "been", "but", "by", "can", "de", "did", "do", "does", "for", "from",
               "had", "has", "have", "he", "her", "him", "his", "how", "if", "in", "into",
               "is", "it", "its", "more", "my", "nbsp", "new", "no", "non", "not", "of",
               "on", "one", "or", "other", "our", "she", "so", "some", "such", "than",
               "that", "the", "their", "then", "there", "these", "they", "this", "those",
               "thus", "to", "up", "us", "use", "was", "we", "were", "what", "when", "where",
               "which", "while", "why", "will", "with", "would", "you", "your", "yours"],
        'de': "_german_",
        'pt': "_portuguese_",
    }

    settings = {
        "number_of_replicas": 0,
        "number_of_shards": 1,
        "refresh_interval": "1m",
        "analysis": {
            "filter": {
                "light_stemmer": {
                    "type": "stemmer",
                    "language": lang_stemmer[lang]
                },
                "lang_stop": {
                    "ignore_case": "true",
                    "type": "stop",
                    "stopwords": lang_stop[lang]
                }
            },
            "char_filter": {
                "icu_normalize": {
                    "type": "icu_normalizer",
                    "name": "nfkc",
                    "mode": "compose"
                }
            },
            "analyzer": {
                "text_no_stop": {
                    "filter": [
                        "lowercase",
                        "light_stemmer"
                    ],
                    "tokenizer": "whitespace" if args.pyizumo_tokenization else "standard",
                    "char_filter": [
                        "icu_normalize"
                    ],
                },
                "text_stop": {
                    "filter": [
                        "lowercase",
                        "lang_stop",
                        "light_stemmer"
                    ],
                    "tokenizer": "whitespace" if args.pyizumo_tokenization else "standard",
                    "char_filter": [
                        "icu_normalize"
                    ],
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
    coga_mappings = {
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
                "analyzer": "text_no_stop",
                "search_analyzer": "text_stop",
                "term_vector": "with_positions_offsets",
                "index_options": "offsets",
                "store": "true"
            },
            "fileTitle": {
                "type": "text",
                "analyzer": "text_no_stop",
                "search_analyzer": "text_stop",
                "term_vector": "with_positions_offsets",
                "index_options": "offsets",
                "store": "true"
            },
            "title_paraphrases": {
                "type": "text",
                "analyzer": "text_no_stop",
                "search_analyzer": "text_stop",
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
                "analyzer": "text_no_stop",
                "search_analyzer": "text_stop",
                "term_vector": "with_positions_offsets",
                "index_options": "offsets",
                "store": "true"
            },
            "plainTextContent": {
                "type": "text",
                "analyzer": "text_no_stop",
                "search_analyzer": "text_stop",
                "term_vector": "with_positions_offsets",
                "index_options": "offsets",
                "store": "true"
            },
            "title_and_text": {
                "type": "text",
                "analyzer": "text_no_stop",
                "search_analyzer": "text_stop",
                "term_vector": "with_positions_offsets",
                "index_options": "offsets",
                "store": "true"
            },
            "app_name": {
                "type": "text",
                "analyzer": "text_no_stop",
                "search_analyzer": "text_stop",
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


def create_es_client(fingerprint, api_key, host):
    # global ES_SSL_FINGERPRINT, ES_API_KEY, client
    ES_SSL_FINGERPRINT = os.getenv(fingerprint)
    ES_API_KEY = os.getenv(api_key)
    _client = Elasticsearch(f"https://{host}:9200",
                            ssl_assert_fingerprint=(ES_SSL_FINGERPRINT),
                            api_key=ES_API_KEY,
                            request_timeout=60
                            )
    return _client


def extract_answers(res):
    rout = []
    for rank, r in enumerate(res.body['hits']['hits']):
        rout.append({'id': r['_id'], 'score': r['_score'], 'text': r['_source']['text']})
    return rout


def build_elastic_query(qid, text, db_engine, model_name, hybrid_mode, model_on_server, vector_field_name):
    _knn = None
    _query = None
    _rank = None
    if db_engine in ["es-dense"]:
        _knn = {
            "field": vector_field_name,
            "k": args.top_k,
            "num_candidates": 1000,
        }
        if model_on_server:
            _knn["query_vector_builder"] = {
                "text_embedding": {
                    "model_id": args.model_name,
                    "model_text": text
                }
            }
        else:
            _knn['query_vector'] = compute_embedding(model, text, args.normalize_embs)

        if hybrid_mode == "rrf":
            _query = {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query_text,
                            "fields": ['text', 'title']
                        }
                    },
                }
            }
            _rank = {"rrf": {
                "window_size": 200
            }}
    elif args.db_engine == 'es-bm25':
        _query = {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": text,
                        "fields": ['text', 'title']
                    }
                },
            }
        }
    elif args.db_engine == "es-elser":
        _query = {
            "text_expansion": {
                "ml.tokens": {
                    "model_id": args.model_name,
                    "model_text": query_text
                }
            }
        }
    return _query, _knn, _rank


if __name__ == '__main__':
    from datetime import datetime

    with open("logfile", "a") as cmdlog:
        cmdlog.write(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {os.getenv('USER')} - "
                     f"{' '.join(sys.argv)}\n")
    parser = setup_argparse()

    args = parser.parse_args()

    if not args.model_name:
        if args.db_engine == 'es-elser':
            args.model_name = ".elser_model_1"
        elif args.db_engine.startswith('es-dense'):
            args.model_name = 'all-MiniLM-L6-v2'

    if args.data_type == "beir":
        if args.input_passages is None:
            args.input_passages = os.path.join(args.data, "corpus.jsonl")
        if args.input_queries is None:
            args.input_queries = os.path.join(args.data, "queries.jsonl")

    server_map = {
        'CONVAI': "convaidp-nlp.sl.cloud9.ibm.com",
        'AILANG': "ai-lang-conv-es.sl.cloud9.ibm.com",
        'local': "localhost",
        'RESCONVAI': "convai.sl.res.ibm.com"
    }

    normalize_text = args.normalize_text

    if args.host is None:
        if args.server == "SAP" or args.server == "SAP_TEST":
            print("The SAP server has been retired. Use CONVAI or AILANG")
            sys.exit(33)
        if args.server in server_map.keys():
            args.host = server_map[args.server]
        else:
            print(f"Server {args.server} is not known!")
            sys.exit(12)

    if args.index_name is None:
        index_name = (
            f"{args.data}_{args.db_engine}_{args.model_name if args.db_engine == 'es-dense' else 'elser' if args.db_engine == 'es-elser' else 'bm25'}_index").lower()
    else:
        index_name = args.index_name.lower()

    index_name = re.sub('[^a-z0-9]', '-', index_name)

    if not nlp:
        nlp = pyizumo.load(args.lang, parsers=['token', 'sentence'])

    do_ingest = 'i' in args.actions
    do_retrieve = 'r' in args.actions
    do_rerank = 'R' in args.actions
    do_update = 'u' in args.actions
    doc_based_ingestion = args.doc_based
    docname2url = {}
    docname2title = {}
    if args.hana_file2url is not None:
        with open(args.hana_file2url) as inp:
            # fl = csv.reader(inp, delimiter="\t")
            for ln in inp.readlines():
                line = ln.strip().split("\t")
                docname2url[line[0]] = line[1]
                docname2title[line[0]] = line[2].strip()

    docid_filter = []
    if args.docids_to_ingest is not None:
        with open(args.docids_to_ingest) as inp:
            for line in inp:
                line = line.replace(".txt", "").strip()
                docid_filter.append(line)

    docid2loio = {}
    if args.docid_map is not None:
        with open(args.docid_map) as inp:
            for line in inp:
                a = line.split()
                docid2loio[a[0]] = a[1]

    model = None
    if args.db_engine in ['es-dense'] or args.max_doc_length is not None:
        import torch

        batch_size = 64

        if args.db_engine == 'es-elser':
            model = MyEmbeddingFunction('all-MiniLM-L6-v2')
        else:
            model = MyEmbeddingFunction(args.model_name)

    print(f"Using the {args.server} server at {args.host}")

    if args.server == "CONVAI":
        client = create_es_client("CONVAI_SSL_FINGERPRINT", "CONVAI_API_KEY", args.host)
    elif args.server == "AILANG":
        client = create_es_client("AILANG_SSL_FINGERPRINT", "AILANG_API_KEY", args.host)
    elif args.server == "RESCONVAI":
        client = create_es_client("RESCONVAI_SSL_FINGERPRINT", "RESCONVAI_API_KEY", args.host)
    elif args.server == "local":
        client = create_es_client("LOCAL_SSL_FINGERPRINT", "LOCAL_API_KEY", args.host)
    else:
        print(f"Server {args.server} is unknown. Exiting..")
        sys.exit(12)

    try:
        res = client.info()
    except Exception as e:
        print(f"Error: {e}")
        raise e

    # client = Elasticsearch("https://localhost:9200",
    #                        ca_certs="/home/raduf/sandbox2/primeqa/ES-8.8.1/elasticsearch-8.8.1/config/certs/http_ca.crt",
    #                        basic_auth=("elastic", ELASTIC_PASSWORD)
    #                        )
    init_settings()
    stopwords = None
    if do_ingest or do_update:
        max_documents = args.max_num_documents

        input_passages = read_data(args.input_passages,
                                   lang=args.lang,
                                   fields=["id", "text", "title"],
                                   remove_url=args.replace_links,
                                   max_doc_size=args.max_doc_length,
                                   stride=args.stride,
                                   tokenizer=model.tokenizer if model is not None else None,
                                   max_num_documents=max_documents,
                                   doc_based=doc_based_ingestion,
                                   docname2url=docname2url,
                                   docname2title=docname2title,
                                   remove_stopwords=args.remove_stopwords,
                                   docid_filter=docid_filter,
                                   uniform_product_name=args.product_name,
                                   data_type=args.data_type,
                                   docid_map=docid2loio,
                                   )
        if max_documents is not None and max_documents > 0:
            input_passages = input_passages[:max_documents]

        hidden_dim = -1
        passage_vectors = []
        if args.db_engine in ['es-dense']:
            print("Encoding corpus documents:")
            passage_vectors = model.encode([passage['text'] for passage in input_passages], _batch_size=batch_size)

            hidden_dim = len(passage_vectors[0])
            if args.normalize_embs:
                passage_vectors = normalize(passage_vectors)

        logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)

        if args.pyizumo_tokenization:
            for passage in tqdm(input_passages, desc="Getting pyizumo word tokens"):
                passage['title'] = get_pyizumo_tokenized_text(text=passage['title'], language_code=args.language_code)
                passage['text'] = get_pyizumo_tokenized_text(text=passage['text'], language_code=args.language_code)

        vector_field_name = None
        if args.db_engine in ['es-dense', 'es-bm25']:
            mappings = coga_mappings[args.lang]
            processors = []
            if args.db_engine in ['es-dense']:
                if args.model_on_server:
                    vector_field_name = "vector"
                    pipeline_name = f"{args.model_name}-test"
                    processors = [{
                        "inference": {
                            "model_id": args.model_name,
                            "target_field": vector_field_name,
                            "field_map": {
                                "text": "text_field"
                            }
                        }
                    }]
                    on_failure = [{
                        "set": {
                            "description": "Index document to 'failed-<index>'",
                            "field": "_index",
                            "value": "failed-{{{_index}}}"
                        }
                    },
                    {
                        "set": {
                            "description": "Set error message",
                            "field": "ingest.failure",
                            "value": "{{_ingest.on_failure_message}}"
                        }
                    }]
                else:
                    vector_field_name = "vector"
                    pipeline_name = None
                    on_failure = None

                mappings['properties'][vector_field_name] = {
                    "type": "dense_vector",
                    "similarity": "cosine",
                    "dims": hidden_dim,
                    "index": "true"
                }

            create_update_index(client, index_name, settings=settings[args.lang],
                                mappings=mappings,
                                do_update=do_update)

            if args.models_on_server and len(processors)>0:
                client.ingest.put_pipeline(processors=processors, id=pipeline_name, on_failure=on_failure)

            logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
            bulk_batch = args.ingestion_batch_size

            num_passages = len(input_passages)
            keys_to_index = ['title', 'id', 'url', 'productId',  # 'versionId',
                             'filePath', 'deliverableLoio', 'text', 'app_name']
            t = tqdm(total=num_passages, desc="Ingesting dense documents: ", smoothing=0.05)
            print(input_passages[0].keys())
            for k in range(0, num_passages, bulk_batch):
                actions = [
                    {
                        "_index": index_name,
                        "_id": row['id'],
                        "_source": {k: row[k] for k in keys_to_index}
                    }
                    for pi, row in enumerate(input_passages[k:min(k + bulk_batch, num_passages)])
                ]
                if args.db_engine in ['es-dense']:
                    if not args.model_on_server:
                        for pi, (action, row) in enumerate(
                                zip(actions, input_passages[k:min(k + bulk_batch, num_passages)])):
                            action["_source"]['vector'] = passage_vectors[pi + k]
                try:
                    bulk(client, actions=actions, pipeline=pipeline_name)
                except Exception as e:
                    print(f"Got an error in indexing: {e}")
                t.update(bulk_batch)
            t.close()
            if len(actions) > 0:
                try:
                    bulk(client=client, actions=actions, pipeline=pipeline_name)
                except Exception as e:
                    print(f"Got an error in indexing: {e}, {len(actions)}")
        elif args.db_engine == "es-elser":
            mappings = coga_mappings[args.lang]
            mappings['properties']['ml.tokens'] = {"type": "rank_features"}

            processors = [
                {
                    "inference": {
                        "model_id": args.model_name,
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
            create_update_index(client,
                                index_name=index_name,
                                settings=settings[args.lang],
                                mappings=mappings,
                                do_update=do_update)

            client.ingest.put_pipeline(processors=processors, id=args.model_name + "-test")
            actions = []
            all_keys_to_index = ['title', 'id', 'url', 'productId',
                                 'filePath', 'deliverableLoio', 'text', 'app_name']
            keys_to_index = []
            for k in all_keys_to_index:
                if k not in input_passages[0]:
                    print(f"Dropping key {k} - they are not in the passages")
                else:
                    keys_to_index.append(k)
            num_passages = len(input_passages)
            t = tqdm(total=num_passages, desc="Ingesting documents (w ELSER): ", smoothing=0.05)
            # for ri, row in tqdm(enumerate(input_passages), total=len(input_passages), desc="Indexing passages"):
            for k in range(0, num_passages, bulk_batch):
                actions = [
                    {
                        "_index": index_name,
                        "_id": row['id'],
                        "_source": {k: row[k] for k in keys_to_index}
                    }
                    for pi, row in enumerate(input_passages[k:min(k + bulk_batch, num_passages)])
                ]

                failures = 0
                while failures < 5:
                    try:
                        res = bulk(client=client, actions=actions, pipeline=args.model_name + "-test")
                        break
                    except Exception as e:
                        print(f"Got an error in indexing: {e}, {len(actions)}")
                    failures += 5
                t.update(bulk_batch)
            t.close()

            if len(actions) > 0:
                try:
                    bulk(client=client, actions=actions, pipeline=args.model_name + "-test")
                except Exception as e:
                    print(f"Got an error in indexing: {e}, {len(actions)}")

        print(f"Product ID histogram:")
        for k in sorted(product_counts.keys(), key=lambda x: product_counts[x], reverse=True):
            print(f" {k}\t{product_counts[k]}")

    ### QUERY TIME

    if do_retrieve:
        loio2docid = {}
        if args.docid_map is not None:
            with open(args.docid_map) as inp:
                for line in inp:
                    a = line.split()
                    loio2docid[a[1]] = a[0]

        if args.evaluate:
            input_queries, unmapped_ids = read_data(args.input_queries,
                                                    lang=args.lang,
                                                    fields=["id", "text", "relevant", "answers"],
                                                    docid_map=loio2docid, return_unmapped=True,
                                                    remove_stopwords=args.remove_stopwords,
                                                    data_type=args.data_type,
                                                    doc_based=doc_based_ingestion)
            print("Unmapped ids:", unmapped_ids)
        else:
            input_queries = read_data(args.input_queries,
                                      lang=args.lang,
                                      fields=["id", "text"],
                                      remove_stopwords=args.remove_stopwords,
                                      data_type=args.data_type,
                                      doc_based=doc_based_ingestion)

        result = []
        for query in tqdm(input_queries):
            qid = query['id']
            query_text = query['text']
            query, knn, rank = build_elastic_query(qid=qid,
                                                   db_engine=args.db_engine,
                                                   text=query_text,
                                                   model_name=args.model_name,
                                                   hybrid_mode=args.hybrid,
                                                   model_on_server=args.model_on_server)
            res = client.search(
                index=index_name,
                knn=knn,
                query=query,
                rank=rank,
                size=args.top_k,
                source_excludes=['vector']
            )
            result.append({'qid': qid, 'text': query_text,
                           "answers": extract_answers(res)})

        if do_rerank:
            pass

        if args.output_file is not None:
            if args.output_file.endswith(".json"):
                with open(args.output_file, 'w', encoding="utf8") as out:
                    json.dump(result, out, indent=2, ensure_ascii=False)
            elif args.output_file.endswith(".jsonl"):
                with open(args.output_file, 'w', encoding="utf8") as out:
                    for r in result:
                        json.dump(r, out, ensure_ascii=False)
                        out.write("\n")

        if args.evaluate:
            score = compute_score(input_queries, result)

            with open(args.output_file.replace(".json", ".metrics"), "w") as out:
                json.dump(score, out, indent=2)
