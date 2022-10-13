# Open Retrieval Question Answering (ORQA)

Given a question,  the retriever component retrieves the supporting documents and a reader component produces the answer conditioned on the supporting documents. 

## Long Form Question Answering (LFQA)

LFQA is a form of generative question answering. The supporting documents are used to generate multi-sentence answers. The following shows how to use the information retrieval and reader components to generate and answer given a question.

- Step 1:  Initialize the retriever. You can choose any of the IR models we currently have ....

```python
retriever = ColBERTRetriever()
# The parameters of the retriever to be determined
retriever.set_parameter_value()
```

- Step 2:  Initialize the reader model. You can choose any the generative QA model we currently have ...

```python
reader = GenerativeFiDReader()
reader.set_parameter_value("model_name_or_path", "PrimeQA/fid_dpr_bart_large")
```

- Step 3:  Initialize the QA pipeline model. 

```python
orqa_pipeline = QAPipeline(retriever, reader)
orqa_pipeline.load()
```

- Step 4:  Execute the generative pipeline in inference mode. 

```python
query="What causes the trail behind jets at high altitude?",
answers = orqa_pipeline.predict(query)
```

The above statements will generate an output in the form of a dictionary:
```shell
{"prediction_text": "The water vapor from the exhaust of the jet is condensing on the surrounding air, which comes from either the surrounding water vapor or the exhaust itself, leaving behind a visible trail. \n\nThe contrails are a manmade type of cirrus cloud formed when water vapor and ice nuclei are frozen and form a large ice sheet."}
```