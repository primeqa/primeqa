from primeqa.util.searchable_corpus import SearchableCorpus, read_tsv_data, compute_score
import os
from tqdm import tqdm

if __name__ == '__main__':
    corpus_name = 'nq910'
    # model_name = "es-bm25"
    model_name = "faiss"
    index_name = f"nlp-{corpus_name}-elser-2023-08-07"
    max_num_questions = -1
    data_dir = f"/home/raduf/sandbox2/ibm-generative-ai-cookbooks/notebooks/data/rag/{corpus_name}"
    passages_file = os.path.join(data_dir, 'psgs.tsv')

    if model_name.startswith("es-"):
        corpus = SearchableCorpus(model_name=model_name, top_k=40, index_name=index_name,
                                  fields=["text", 'title'], server="https://9.59.196.68:9200")
    elif model_name == 'faiss':
        corpus = SearchableCorpus(model_name=model_name, top_k=40, data_id=passages_file,
                                  emb_model_name="sentence-transformers/all-mpnet-base-v2")
        #corpus.pool = corpus.emb_model.encode_multi_process()
        corpus.start_pool()

    else:
        corpus = SearchableCorpus(model_name=model_name, top_k=40, data_id=passages_file)

    passages = read_tsv_data(input_file=passages_file, fields=['id', 'text', 'title'])

    if not model_name.startswith("es-"):
        if model_name=="faiss":
            corpus.add(texts=passages)
            corpus.stop_pool()
        else:
            corpus.add(texts=passages_file)

    questions = read_tsv_data(input_file=os.path.join(data_dir, "questions.tsv"),
                          fields=["id", "text", "relevant", "answers"])

    if max_num_questions>=0:
        questions = questions[:max_num_questions]

    # answers, scores = corpus.search(input_queries=questions)

    answers = []
    for qid, question in tqdm(enumerate(questions), total=len(questions), desc="Processing questions: "):
        response = corpus.search(input_queries=[question['text']])
        # ans = []
        # for rank, match in enumerate(response[0]):
        #     ans.append(match)
        #     # answers.append([questions[qid]['id'], match, rank + 1, response[0][rank]])
        #     # ans.append(match)

        answers.append(response[0])

    score = compute_score(input_queries = questions, input_passages=passages, answers=answers)

    print(score)