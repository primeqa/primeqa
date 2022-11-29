## Run ORQA 

This will run a Retriever-Retriever pipeline given a set of questions.

### Prerequisites

Before continuing below make sure you have PrimeQA [installed](../../README.md#Installation).

Additionally, install `ibm_watson` package via pip if using IBM® Watson Discovery as the Retriever.

```
pip install ibm_watson

```

Before configuring the pipeline, you will need to have available a collection index and a reader model. 

- To use the IBM® Watson Discovery retriever, first configure a IBM® Watson Discovery Cloud instance using the instructions [here](https://cloud.ibm.com/catalog/services/watson-discovery) and create a collection index.

- To use the PrimeQA retriever, first setup the collection index for the Retriever using the instructions [here](https://github.com/primeqa/primeqa/tree/main/primeqa/ir/).


You will also need a PrimeQA reader model. You can use one from Huggingface model hub, for example, `PrimeQA/nq_tydi_sq1-reader-xlmr_large-20221110`.  To train or fine-tune a model on your data, please follow the instructions [here](https://github.com/primeqa/primeqa/tree/main/examples/custom_mrc/)


Once you have an collection index and a reader, please continue with the configuration.

### Configuration 

The configuration examples below show how to set up a Retriever and Reader. Additionally, the `reranking` section allows tuning of how the scores from the `retriever` and `reader` are combined to obtain a final score for ranking the answers. The two scores are normalized using the Frobenius and combined using a convex combination. The weight in the combination operation applied to the `retriever` score is specified in the `ir_weight` field under `reranking` and is typically obtained by tuning on a validation set.  This combined score is used to do a final ranking of the answers. 


#### Discovery Retriever

```
    {
        "retrievers" : [
            {
                "name": "WatsonDiscoveryRetriever",
                "endpoint": "<IBM® Watson Discovery Cloud/CP4D Instance Endpoint>",
                "apikey": "<API key (If using IBM® Watson Discovery Cloud instance)>",
                "project_id": "<IBM® Watson Discovery Project ID>",
                "index_name": "<collection-name>",
                "max_num_documents": 5
            }
        ],
        "reader" : {
            "name": "ExtractiveReader",
            "model_name_or_path": "PrimeQA/nq_tydi_sq1-reader-xlmr_large-20221110",
            "max_answer_length": 64,
            "max_num_answers": 10
        }
        "reranking": {
            "method" : "weighted_average_ir_mrc"
            "ir_weight": <final score combination , a value between 0 and 1>
        }
    }

```

#### PrimeQA Retriever

```

    {
        "retrievers" : [
            {
                "name": "ColBERTRetriever",
                "checkpoint": "<path-to-colbert-model-file>",
                "index_root": "index-root-dir",
                "index_name": "index-subdirectory",
                "max_num_documents": 5,
                "corpus_tsv_file_path": "<corpus-tsv-file>"
            }
        ],
        "reader" : {
            "name": "ExtractiveReader",
            "model_name_or_path": "PrimeQA/nq_tydi_sq1-reader-xlmr_large-20221110",
            "max_answer_length": 64,
            "max_num_answers": 10
        }
        "reranking": {
            "method" : "weighted_average_ir_mrc"
            "ir_weight": <score combination weight assigned to IR, a value between 0 and 1>
        }
    }

```


