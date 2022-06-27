import os
import pickle
from re import X
from typing import Any, List

import marisa_trie
import neuralcoref
# import pandas as pd
import spacy
import torch
# from datasets import load_dataset
# from openie import StanfordOpenIE
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.pipeline import EntityRuler
from spacy.tokens import Token
from tqdm import tqdm
# from transformers import RagRetriever, RagTokenForGeneration, RagTokenizer

# import KID
 
# ipywidgets

stop_words_getter = lambda token: token.is_stop or token.lower_ in STOP_WORDS \
                                                or token.lemma_ in STOP_WORDS

Token.set_extension('is_stop', getter=stop_words_getter, force=True)

nlp = spacy.load("en_core_web_sm")
ruler = EntityRuler(nlp)
nlp.add_pipe(ruler)

neuralcoref.add_to_pipe(nlp)

# properties = {
#     'openie.affinity_probability_cap': 1 / 3,
# }

# retriever_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
# retriever = RagRetriever.from_pretrained(
#     "facebook/rag-token-nq", index_name="legacy", use_dummy_dataset=False
# )
# rag_model = RagTokenForGeneration.from_pretrained(
#     "facebook/rag-token-nq", retriever=retriever
# )

# kg_dataset = load_dataset("wiki_dpr", 'psgs_w100.nq.exact')


def return_retrieved_psgs(
    tokenizer: Any, retriever: Any, rag: Any, query: str, k_docs: int
):
    """Retrieve docs by query.

    :param tokenizer: the tokenizer of RAG
    :param rag: the RAG model loaded
    :param retriever: the retriever of RAG
    :param query: the question/context of the NLG task. A string.
    :param k_docs: the number of documents to be retrieved
    :return: two lists. [doc ids] and [doc scores].
    """
    inputs = tokenizer(query, return_tensors="pt")
    input_ids = inputs["input_ids"]

    question_hidden_states = rag.question_encoder(input_ids)[0]
    docs_dict = retriever(
        input_ids.numpy(),
        question_hidden_states.detach().numpy(),
        return_tensors="pt",
        n_docs=k_docs
    )
    doc_scores = torch.bmm(
        question_hidden_states.unsqueeze(1),
        docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
    ).squeeze(1)
    return docs_dict['doc_ids'].detach().cpu().numpy().tolist()[0], doc_scores.detach(
    ).cpu().numpy().tolist()[0]


def convert2kg(texts: List[str], client: Any):
    """Extract <subj., rel., obj.> triplets from list of texts.

    :param texts: a list of text
    :param client: the CoreNLPClient openie client
    :return:
    """
    # str_list = []
    # for text in texts:
    #     resolved_text = nlp(text)._.coref_resolved
    #     for triple in client.annotate(resolved_text):
    #         str_list.append(triple['subject'] + ' ' + triple['object'])
    # return str_list

    str_list = set()
    for text in texts:
        resolved_text = nlp(text)._.coref_resolved
        doc = client.annotate(resolved_text)
        for sent in doc.sentence:
            # https://github.com/stanfordnlp/stanza/blob/011b6c4831a614439c599fd163a9b40b7c225566/doc/CoreNLP.proto#L558
            for triple in sent.openieTriple:
                s = triple.subject + ' ' + triple.object
                score = triple.confidence
                rel = triple.relation
                # s = ' '.join(return_entities(s)) # FIXME The original KID code doesn't do preprocessing but the paper mentioned (lower, preprocessing, etc) 
                str_list.add(s.lower())
    return list(str_list)



def return_entities(sent: str):
    return [
        token.lemma_.lower() for token in nlp(sent)
        if token.pos_ in ['PROPN', 'NOUN'] and not token.is_stop
    ]



def load_eli5_dpr(dataset_fn, n_docs):
    import json
    examples = []
    with open(dataset_fn) as f:
        for line in f:
            ex = {}
            data = json.loads(line.strip())
            ex["id"] = data["id"]
            ex["input"] = data["input"]
            ex["passages"] =  [x["text"] for x in data["passages"]][:n_docs]
            examples.append(ex)
    return examples

import os
from stanza.server import CoreNLPClient
os.environ['CORENLP_HOME'] = '/dccstor/myu/.stanfordnlp_resources/stanford-corenlp-4.4.0'

def create_external_graph_eli5(n_docs, task_path, examples, start=None, end=None):
    print("start and end", start, end)
    examples = examples[start:end]
    # else take all examples
    full_str_list = []
    with CoreNLPClient(annotators=['openie'], 
                   memory='8G', endpoint='http://localhost:1090', be_quiet=True) as client:
        for ex in tqdm(examples, total=len(examples)):
            kg_docs_text = ex["passages"] # a list of strings
            full_str_list.extend(convert2kg(kg_docs_text, client))

    with open (task_path + f'/full_str_{n_docs}_{start}-{end}.txt', 'w') as fw:
        for line in full_str_list:
            fw.write(line + '\n')

    # trie = marisa_trie.Trie(full_str_list)
    # with open(task_path + '/trie_' + str(n) + '.pickle', 'wb') as handle:
    #     pickle.dump(trie, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    n=3
    examples = load_eli5_dpr(
        "/dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr.json",
        n_docs=n
    )   
    output_path = "/dccstor/myu/experiments/knowledge_trie/openie_dev"
    assert os.path.isdir(output_path)
    import sys
    args = sys.argv
    assert len(args) == 3
    create_external_graph_eli5(n, output_path, examples, start=int(args[1]), end=int(args[2]))