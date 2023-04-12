
from primeqa.ir.dense.colbert_top.colbert.infra.run import Run
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig, RunConfig
from primeqa.ir.dense.colbert_top.colbert import Trainer

def distill_triples_with_retriever(given_triples, given_synth_queries, given_checkpoint, chosen_split, chosen_type, chosen_set, dataset, chosen_BEIR_set, chosen_BEIR_type, downloads_folder):
    
    nranks = torch.cuda.device_count()
    
    with Run().context(RunConfig(nranks=nranks)):
        
        if dataset == "LoTTE":
        	collection = downloads_folder + '/lotte/' + chosen_split + '/' + chosen_set +'/collection.tsv'
        elif dataset == "BEIR":
        	collection = downloads_folder + '/beir_datasets/' + chosen_BEIR_set + '/' + chosen_BEIR_type +'/collection.tsv'
        else:
            raise ValueError("Incorrect dataset specification for triples")

        config = ColBERTConfig(bsize=16, lr=1e-05, warmup=None, doc_maxlen=300, dim=128, nway=16, accumsteps=2, use_ib_negatives=False)
        trainer = Trainer(triples=given_triples, queries=given_synth_queries, collection=collection, config=config)

        distilled_checkpoint = trainer.train(checkpoint=given_checkpoint)

        print("Generated distilled checkpoint!")
        print(distilled_checkpoint)

        return distilled_checkpoint


