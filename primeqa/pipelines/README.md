<!-- START sphinx doc instructions - DO NOT MODIFY next code, please -->
<details>
<summary>API Reference</summary>    

```{eval-rst}

.. autosummary::
    :toctree: _autosummary
    :template: custom-module-template.rst
    :recursive:
   
    primeqa.pipelines

```
</details>          
<br>
<!-- END sphinx doc instructions - DO NOT MODIFY above code, please --> 

# QA Pipelines

The pipeline allows the components to be used as building blocks and allows switching out alternative implementations.  

We show some pipeline examples for the Open Retrieval Question Answering. Open retrieval systems query large document stores for relevant passages. These passages are used by the reader to produce the answer.

## Open Retrieval with Extractive Reader

The pipeline uses a ColBERT retriever to obtain supporting passages. The reader extracts the best answers from the passages and provides evidence for its predictions.

- Step 1:  Initialize the retriever.

```python
retriever = ColBERTRetriever(index_root = index_root, index_name = index_name, collection = collection, max_num_documents = 3)
retriever.load()
```

- Step 2:  Initialize the reader model. 

```python
reader = ExtractiveReader(model_name_or_path = model)
reader.load()
```

- Step 3:  Initialize the QA pipeline. 

```python
extractive_pipeline = QAPipeline(retriever, reader)
```

- Step 4:  Execute the pipeline in inference mode. 

```python
queries=["Which country is Canberra located in?"]
answers = pipeline.run(query)
```

## Long Form Question Answering (LFQA) 

LFQA is a generative task where the retrieved passages are used to generate a complex multi-sentence answer.

In this example we show a QA Pipeline using a ColBERT retriever and a Fusion in Decoder (FID) generator.

Instructions to create a ColBERT index and an FiD model for KILT-ELI5 can be found [here](https://github.com/primeqa/primeqa/blob/main/examples/lfqa/README.md)

- Step 1:  Initialize the retriever.

```python
retriever = ColBERTRetriever(index_root = index_root, index_name = index_name, collection = collection, max_num_documents = 3)
retriever.load()
```

- Step 2:  Initialize the reader model. 

```python
reader = GenerativeFiDReader(model_name_or_path = model)
reader.load()
```

- Step 3:  Initialize the QA pipeline. 

```python
lfqa_pipeline = QAPipeline(retriever, reader)
```

- Step 4:  Execute the LFQA pipeline in inference mode. 

```python
queries=["What causes the trail behind jets at high altitude?"]
answers = lfqa_pipeline.run(query)
```