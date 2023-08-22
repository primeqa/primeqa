from primeqa.util.searchable_corpus import SearchableCorpus, read_tsv_data, compute_score
import os
from tqdm import tqdm

corpus = SearchableCorpus(model_name="bm25", top_k=20)

data_dir = "/home/raduf/sandbox2/ibm-generative-ai-cookbooks/notebooks/data/rag/nq910"

passages_file = os.path.join(data_dir, 'psgs.tsv')
passages = read_tsv_data(input_file=passages_file, fields=['id', 'text', 'title'])

corpus.add(texts=passages_file)

questions = read_tsv_data(input_file=os.path.join(data_dir, "questions.tsv"),
                      fields=["id", "text", "relevant", "answers"])

# answers, scores = corpus.search(input_queries=questions)

answers = []
for qid, question in tqdm(enumerate(questions), total=len(questions), desc="Processing questions: "):
    q_ids, response = corpus.search(input_queries=[question['text']])
    ans = []
    for rank, match in enumerate(q_ids[0]):
        # answers.append([questions[qid]['id'], match, rank + 1, response[0][rank]])
        ans.append(match)
    answers.append(ans)

score = compute_score(input_queries = questions, input_passages=passages, answers=answers)

print(score)