from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig, RunConfig

from primeqa.ir.dense.colbert_top.colbert.utils.parser import Arguments
from primeqa.ir.dense.colbert_top.colbert.infra import Run, RunConfig
from primeqa.ir.dense.colbert_top.colbert.indexer import Indexer

def main():
    parser = Arguments(description='Training ColBERT with <query, positive passage, negative passage> triples.')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_indexing_input()
    parser.add_compressed_index_input()

    parser.add_argument('--kmeans_niters', dest='kmeans_niters', default=20, type=int)
    parser.add_argument('--nway', dest='nway', default=2, type=int)

    args = parser.parse()

    # Namespace to dict
    args_dict = vars(args)
    # remove keys not in ColBERTConfig
    args_dict = {key: args_dict[key] for key in args_dict if key not in ['run', 'nthreads', 'distributed', 'compression_level', 'input_arguments']}
    # args_dict to ColBERTConfig
    colBERTConfig = ColBERTConfig(**args_dict)

    with Run().context(RunConfig(root=args.root, experiment=args.experiment, nranks=args.nranks, amp=args.amp)):
        indexer = Indexer(args.checkpoint, colBERTConfig)
        indexer.index(name=args.index_name, collection=args.collection, overwrite=True)


if __name__ == "__main__":
    main()
