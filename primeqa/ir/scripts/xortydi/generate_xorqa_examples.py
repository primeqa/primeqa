#!/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import json
import jsonlines
import numpy.random
from tqdm import tqdm
from primeqa.ir.sparse.retriever import PyseriniRetriever
import unicodedata
import pandas as pd
import os
from utils import HasAnswerChecker

numpy.random.seed(1234)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def handle_args():
    usage = "usage"
    parser = argparse.ArgumentParser(usage)
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="XORQA training data json containing positive_ctxs and negative_ctxs",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="output directory"
    )
    parser.add_argument(
        "--question_translations_dir",
        type=str,
        required=True,
        help="XORTyDI released human translation of training questions",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        required=True,
        help="BM25 (Pyserini) index of wikipedia 100 token passages",
    )
    parser.add_argument(
        "--max_num_ir_based_negatives",
        type=int,
        default=300,
        help="num negative bm25 hits",
    )
    parser.add_argument(
        "--max_num_ir_based_positives",
        type=int,
        default=10,
        help="num positive bm25 hits",
    )
    parser.add_argument(
        "--num_rounds", type=int, default=1, help="num sets of triples to output"
    )
    parser.add_argument(
        "--max_num_negatives",
        type=int,
        default=100,
        help="num negative passages to output",
    )
    parser.add_argument(
        "--max_num_positives",
        type=int,
        default=3,
        help="num positive passages to output",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        default=True,
        help="random shuffle of examples",
    )
    parser.add_argument(
        "--add_title_text",
        action="store_true",
        default=True,
        help="prepend title to passage",
    )
    parser.add_argument(
        "--do_not_run_match_in_title",
        action="store_true",
        default=False,
        help="do not match answer in title",
    )

    args = parser.parse_args()
    logger.info(vars(args))
    return args


def init_question_translations(question_translations_dir):
    question_translations = {}
    # this is very XOR-TyDi QA specific
    langs = set(["ar", "bn", "fi", "ja", "ko", "ru", "te"])

    for lang in langs:
        with open(f"{question_translations_dir}/{lang}-en/en.txt") as f:
            en_content = f.readlines()
        with open(f"{question_translations_dir}/{lang}-en/{lang}.txt") as f:
            ne_content = f.readlines()

        assert len(en_content) == len(ne_content)
        for ne, en in zip(ne_content, en_content):
            question_translations[ne.strip()] = en.strip()
    return question_translations


def run_query(query, searcher, max_retrieved):
    return searcher.retrieve(query, max_retrieved)


def read_jsonlines(file_name):
    lines = []
    print("loading examples from {0}".format(file_name))
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def clean_text(x):
    return unicodedata.normalize("NFD", x).replace("\n", " ")


def get_text_from_element(el):
    if type(el) == str:
        return clean_text(el)
    elif type(el) == list:
        return clean_text(el[0])


def add_ir_positives_negatives(
    qas, max_num_positives, max_num_negatives, add_title_text
):
    subset_poss = list(range(len(qas)))
    num_queries = 0
    num_records = 0

    for qnum in tqdm(subset_poss):
        qa = qas[qnum]
        q = qa["question"]
        q_en = qa["question_en"]
        p = qa["positive_ctxs"]
        n = qa["negative_ctxs"]
        hn = qa["hard_negative_ctxs"]

        num_queries += 1

        p.extend(qa["ir_positive_ctxs"])
        hn.extend(qa["ir_negative_ctxs"])

        for pos_pos in range(len(p)):
            num_negs_out = 0
            for neg_pos in numpy.random.permutation(len(hn)):
                to_yield = {
                    "question": clean_text(q),
                    "question_en": clean_text(q_en),
                    "np": len(p),
                    "nn": len(n),
                    "nhn": len(hn),
                    "p": get_text_from_element(p[pos_pos]["title"])
                    + " "
                    + get_text_from_element(p[pos_pos]["text"])
                    if add_title_text
                    else get_text_from_element(p[pos_pos]["text"]),
                    "n": get_text_from_element(hn[neg_pos]["title"])
                    + " "
                    + get_text_from_element(hn[neg_pos]["text"])
                    if add_title_text
                    else get_text_from_element(hn[neg_pos]["text"]),
                }
                yield to_yield
                num_records += 1
                num_negs_out += 1

                if max_num_negatives > 0 and num_negs_out >= max_num_negatives:
                    break
            if max_num_positives > 0 and pos_pos >= max_num_positives - 1:
                break


def convert_json_to_df(
    qas, max_num_positives=3, max_num_negatives=100, add_title_text=True
):
    df = pd.DataFrame.from_records(
        add_ir_positives_negatives(
            qas,
            max_num_positives=max_num_positives,
            max_num_negatives=max_num_negatives,
            add_title_text=add_title_text,
        )
    )
    return df


def write_examples_as_triples(
    qas,
    max_num_positives,
    max_num_negatives,
    add_title_text,
    num_rounds,
    randomize,
    output_dir,
):

    df = convert_json_to_df(
        qas,
        max_num_positives=max_num_positives,
        max_num_negatives=max_num_negatives,
        add_title_text=add_title_text,
    )

    cols = ["question", "p", "n"]
    cols_en = ["question_en", "p", "n"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_rounds = 1 if not randomize else num_rounds

    filename = os.path.join(
        output_dir, f"xorqa_triples_3poss_100neg_{num_rounds}ep_rand{randomize}.tsv"
    )
    f = open(filename, "w", encoding="utf-8")
    filename = os.path.join(
        output_dir, f"xorqa_triples_3poss_100neg_en_{num_rounds}ep_rand{randomize}.tsv"
    )
    f_en = open(filename, "w", encoding="utf-8")

    if randomize:
        indexes = []
        for i in range(0, num_rounds):
            perm = numpy.random.permutation(len(df)) if randomize else None
            indexes.extend(perm)

        logger.info(
            f"Writing num rounds {num_rounds}, {len(indexes)} randomized triples to {f.name}"
        )
        df[cols].iloc[indexes].to_csv(f, sep="\t", index=None, header=None)
        logger.info(
            f"Writing num rounds {num_rounds}, {len(indexes)} randomized triples to {f_en.name}"
        )
        df[cols_en].iloc[indexes].to_csv(f_en, sep="\t", index=None, header=None)
    else:
        logger.info(f"Writing {len(df.values)}  triples to {f.name}")
        df[cols].to_csv(f, sep="\t", index=None, header=None)
        logger.info(f"Writing {len(df.values)} triples to {f_en.name}")
        df[cols_en].to_csv(f_en, sep="\t", index=None, header=None)

    logger.info(f"Triples written to {output_dir}")


def run_bm25_retrieval(searcher, answer_checker, qas, question_translations, args):
    if "ir_negative_ctxs" in qas[0] or "ir_positive_ctxs" in qas[0]:
        logger.info("input contains bm25 retrieval contexts. Skipping retrieval")
        return

    logger.info(f"Running retrieval on {len(qas)} queries")
    for qnum, qa in enumerate(tqdm(qas)):
        q = qa["question"]
        p = qa["positive_ctxs"]
        n = qa["negative_ctxs"]
        hn = qa["hard_negative_ctxs"]

        en_q = question_translations[q]
        qa["question_en"] = en_q

        retrieved = run_query(
            en_q, searcher, args.max_num_ir_based_negatives * 2
        )  # assuming 50/50 positive/negative ratio

        negatives = []
        positives = []

        positive_ctxs_texts = [ctxt["text"] for ctxt in p]

        for hit in retrieved:
            title = hit["title"]
            text = hit["text"]
            string_to_search = (
                text if args.do_not_run_match_in_title else title + " " + text
            )
            if answer_checker.has_answer(qa["answers"], string_to_search):
                if len(positives) < args.max_num_ir_based_positives and not max(
                    [ctxt in text for ctxt in positive_ctxs_texts]
                ):
                    positives.append({"title": title, "text": text})
            else:
                if len(negatives) < args.max_num_ir_based_negatives:
                    negatives.append({"title": title, "text": text})

            if (
                len(positives) >= args.max_num_ir_based_positives
                and len(negatives) >= args.max_num_ir_based_negatives
            ):
                break

        qa["ir_negative_ctxs"] = negatives
        qa["ir_positive_ctxs"] = positives

        logger.debug(
            f"{en_q} adding { len(qa['ir_negative_ctxs']) } IR based negatives { len(qa['ir_positive_ctxs']) } IR based positives"
        )

    output_file = os.path.join(args.output_dir, "xortydi_ir_negs_poss.json")
    logger.info(f"writing {output_file}")
    with open(output_file, "w") as out_f:
        json.dump(qas, out_f, indent=4, ensure_ascii=True)


def main():
    args = handle_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    answer_checker = HasAnswerChecker()

    searcher = PyseriniRetriever(args.index_path)

    question_translations = init_question_translations(args.question_translations_dir)

    qas = json.load(open(args.input_file))

    run_bm25_retrieval(searcher, answer_checker, qas, question_translations, args)

    write_examples_as_triples(
        qas,
        args.max_num_positives,
        args.max_num_negatives,
        args.add_title_text,
        args.num_rounds,
        args.randomize,
        args.output_dir,
    )

    logging.info("Success...")


# do main
if __name__ == "__main__":
    main()
