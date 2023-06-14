from primeqa.util import SearchableCorpus, read_tsv_data, compute_score
from argparse import ArgumentParser
import json

def select_column(data, col):
    return [d[col] for d in data]

if __name__ == "__main__":
    p = ArgumentParser(description="Script to test the new SearchableCorpus")
    p.add_argument("-m", "--model", required=True, help="The model to test.")
    p.add_argument("-p", "--psgs", required=True, help="The passages to read")
    p.add_argument("-q", "--queries", required=True, help="The queries to test")
    p.add_argument("-v", "--verbose", action="store_true", help="Will print the query output")
    p.add_argument("--colbert_index", type=str, default=None, help="The ColBERT index, if already created.")

    args = p.parse_args()

    passages = read_tsv_data(args.psgs, fields=['id', 'text', 'title'])
    queries = read_tsv_data(args.queries, fields=['id', 'text', 'relevant', 'answers'])

    collection = SearchableCorpus(model_name=args.model, batch_size=64, top_k=10,
                                  colbert_index=args.colbert_index)

    collection.add(select_column(passages, 'text'),
                   select_column(passages, 'title'),
                   select_column(passages, 'id'))
    # or you can do this:
    # collection.add(args.psgs)
    res, scores = collection.search(select_column(queries, 'text'))
    if args.verbose:
        with open("res.out", "w") as out:
            for q in range(len(queries)):
                for rank, (ans, score) in enumerate(zip(res[q], scores[q])):
                    out.write("\t".join([queries[q]['id'], str(ans), str(rank+1), str(score)])+"\n")
    answers = []

    score = compute_score(queries, passages, res, [1,3,5,10], args.verbose)
    print(f"Score is: {json.dumps(score, indent=2)}")