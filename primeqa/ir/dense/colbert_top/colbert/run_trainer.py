from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig

from primeqa.ir.dense.colbert_top.colbert.utils.parser import Arguments
from primeqa.ir.dense.colbert_top.colbert.infra import Run, RunConfig
from primeqa.ir.dense.colbert_top.colbert.trainer import Trainer

def main():
    parser = Arguments(description='Training ColBERT with <query, positive passage, negative passage> triples.')

    parser.add_model_parameters()
    parser.add_model_training_parameters()
    parser.add_training_input()
    # parser.add_argument('--model_type', dest='model_type', default='bert')
    # comment out as we define the argument at model training parameters

    args = parser.parse()

    assert args.bsize % args.accumsteps == 0, ((args.bsize, args.accumsteps),
                                               "The batch size must be divisible by the number of gradient accumulation steps.")
    assert args.query_maxlen <= 512
    assert args.doc_maxlen <= 512

    args.lazy = args.collection is not None

    # Namespace to dict
    args_dict = vars(args)
    # remove keys not in ColBERTConfig
    # args_dict = {key: args_dict[key] for key in args_dict if key not in ['run', 'lazy', 'nthreads', 'distributed', 'resume_optimizer', 'model_type','input_arguments']}
    args_dict = {key: args_dict[key] for key in args_dict if key not in ['run', 'lazy', 'nthreads', 'distributed', 'input_arguments']}
    # args_dict = {key: args_dict[key] for key in args_dict if key not in ['run', 'lazy', 'nthreads', 'distributed']}
    colBERTConfig = ColBERTConfig(**args_dict)

    with Run().context(RunConfig(root=args.root, experiment=args.experiment, nranks=args.nranks, amp=args.amp)):
        trainer = Trainer(args.triples, args.queries, args.collection, colBERTConfig)
        trainer.train(args.checkpoint)


if __name__ == "__main__":
    main()
