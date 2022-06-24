from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig, RunConfig

from primeqa.ir.dense.colbert_top.colbert.utils.parser import Arguments
from primeqa.ir.dense.colbert_top.colbert.infra import Run, RunConfig
from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher

def main():
    parser = Arguments(description='run ')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_compressed_index_input()
    parser.add_ranking_input()
    parser.add_retrieval_input()

    # parser.add_argument('--ranks_fn', dest='ranks_fn', required=True)
    # parser.add_argument('--topk', dest='topK', default=1000)

    args = parser.parse()

    # Namespace to dict
    args_dict = vars(args)
    # remove keys not in ColBERTConfig
    # args_dict = {key: args_dict[key] for key in args_dict if key not in ['run', 'nthreads', 'distributed', 'compression_level', 'qrels', 'partitions', 'retrieve_only', 'ranks_fn', 'topK', 'input_arguments']}
    # need to keep ranks_fn and topK arguments to save the ranking results
    args_dict = {key: args_dict[key] for key in args_dict if key not in ['run', 'nthreads', 'distributed', 'compression_level', 'qrels', 'partitions', 'retrieve_only', 'input_arguments']}
    # args_dict to ColBERTConfig
    colBERTConfig = ColBERTConfig(**args_dict)

    with Run().context(RunConfig(root=args.root, experiment=args.experiment, nranks=args.nranks, amp=args.amp)):
        searcher = Searcher(args.index_name, checkpoint=args.checkpoint, collection=args.collection, config=colBERTConfig)

        rankings = searcher.search_all(args.queries, args.topK)
        out_fn = args.ranks_fn
        rankings.save(out_fn)

if __name__ == "__main__":
    main()
