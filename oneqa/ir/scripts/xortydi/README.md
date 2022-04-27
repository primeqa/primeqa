## Sparse retrieval

Sparse retrieval is based on BM25 ranking using bag of words representation. It is built on Pyserini which is built on Lucene.  

The ```PyseriniRetriever``` class provides the entry point for running queries against an index.

The instructions below are for creating an index of English Wikipedia passage and use the index to search 
and evaluate performance on the Google translation of the XORTyDI DEV set queries. 

### Java SDK Dependency
Pyserini requires Java 11

### PyseriniRetriever usage
Here's how to run a search query against an index and retrieve ranked list of documents:


```
from oneqa.ir.sparse.retriever import PyseriniRetriever

index_path='<path-to-wikipedia-passage-index>
searcher = PyseriniRetriever(index_path, use_bm25=True, k1=0.9, b=0.4)

query = 'What is the largest region of Germany?'
top_n = 5

hits = searcher.retrieve(query,top_n)

for hit in enumerate(hits):
   print(f"{hit['rank']} {hit['passage_id']} {hit['score']}  {hit['title']} {hit['text']}")
```

Output:

```
0 9135762 9.3326997756958  Lenggries Lenggries Lenggries (Central Bavarian: "Lenggrias") is a municipality in Bavaria, Germany. Lenggries is the center of the Isarwinkel, the region along the Isar between Bad Tölz and Wallgau. The town has about 9,500 inhabitants. By area, it is the largest rural municipality ("Gemeinde") in what was formerly West Germany, and the 7th-largest overall. (All six currently larger "Gemeinden" are in Brandenburg.) The name Lenggries is derived from "Lenngengrieze" (long Gries), a long rubble field with deposits of debris from the bed of the Isar. Lenggries sits on the Isar River before it transitions into the Alpine foothills. To the east
1 9135765 9.332698822021484  Lenggries Oberlandbahn (BOB). Lenggries Lenggries (Central Bavarian: "Lenggrias") is a municipality in Bavaria, Germany. Lenggries is the center of the Isarwinkel, the region along the Isar between Bad Tölz and Wallgau. The town has about 9,500 inhabitants. By area, it is the largest rural municipality ("Gemeinde") in what was formerly West Germany, and the 7th-largest overall. (All six currently larger "Gemeinden" are in Brandenburg.) The name Lenggries is derived from "Lenngengrieze" (long Gries), a long rubble field with deposits of debris from the bed of the Isar. Lenggries sits on the Isar River before it transitions into the Alpine foothills. To
2 16887558 9.208499908447266  Würzburger Stein Würzburger Stein Würzburger Stein is a vineyard in the German wine region of Franconia that has been producing a style of wine, known as "Steinwein" since at least the 8th century. Located on a hill overlooking the Main river outside the city of Würzburg, the vineyard is responsible for what may have been the oldest wine ever tasted. In addition to being one of Germany's oldest winemaking sites, at 85 hectares (210 acres), the vineyard is also one of Germany's largest individual plots. Today the vineyard is one of the warmest sites in the Franconia wine region and is planted
3 3608379 8.919300079345703  Melle, Germany Melle, Germany Melle is a city in the district of Osnabrück, Lower Saxony, Germany. The city corresponds to what used to be the district of Melle until regional territorial reform in 1972. Since then Melle is the third largest city in Lower Saxony in terms of surface area. Melle was first mentioned in a document from 1169. In 1443 Heinrich von Moers, Bishop of Osnabrück, gave Melle the privilege of a "Wigbold". Osnabrück looked after Melle's interests in the Westphalian Hanseatic League. Melle belonged to the Kingdom of Hanover until 1866 when it became part of Prussia. In 1885 Amt
4 3608383 8.919299125671387  Melle, Germany observation. Melle, Germany Melle is a city in the district of Osnabrück, Lower Saxony, Germany. The city corresponds to what used to be the district of Melle until regional territorial reform in 1972. Since then Melle is the third largest city in Lower Saxony in terms of surface area. Melle was first mentioned in a document from 1169. In 1443 Heinrich von Moers, Bishop of Osnabrück, gave Melle the privilege of a "Wigbold". Osnabrück looked after Melle's interests in the Westphalian Hanseatic League. Melle belonged to the Kingdom of Hanover until 1866 when it became part of Prussia. In 1885
```


### Create an Pyserini index of Wikipedia passages for XORTyDI

1. Download the DPR corpus of English Wikpedia (December 20, 2018 dump) split 100 word passages 
   wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
2. Format as JSON
    python convert_corpus_tsv_to_pyserini_jsonl.py --input <psgs_w100_file> --output <output_dir>
3. Build the Pyserini index
   ```
   python -m pyserini.index.lucene --collection JsonCollection --input <psgs_w100_jsonl-dir> --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw --index <index-dir>
   ```

### Run BM25 search 
1. Download the XORTyDI dev set xor_retrieve_eng_span_dev_google_trans_q.jsonl queries file containing quereies translated to English from here: https://drive.google.com/file/d/1JzlNDijDZmDlT42ABVJK53gwk7_mKHGt/view?usp=sharing. 
2. Convert to ColBERT tsv format <id>\t<query>  (TODO: need script)
3. Run:
  ```shell
   python /dccstor/bsiyer6/OneQA/OneQA/examples/ir/run_bm25_retrieval.py \
   --output_dir <output-dir> --index_path <path-to-en-psgs_w100_index> \
   --queries_file <xortydi-dev-gmt-queries.tsv> --top_k 1000 \
   --max_hits_to_output 100 --xorqa_data_file <xortydi-queries-jsonl>
  ```
4. This will produce 2 files in the output directory:
   - ranking.tsv in ColBERT ranking output format
   - ranking_xortydi_format.json in XORTyDI format

### Evaluate XORTyDI BM25 ranked results
Evaluate BM25 ranked results on XORTyDI dev set queries:
1. Clone the XORTyDI repo here: https://github.com/AkariAsai/XORQA
2. Download DEV gold data file here: https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl
3. Run:
   ```shell
   python <path-to-xorqa-repo>/evals/eval_xor_retrieve.py --data_file <path-to-gold-data> --pred_file <path-to-ranking_xortydi_format.json> 
   ```
4. The eval script output should match the following: 
```
Evaluating R@2kt
performance on te (238 examples)
53.78151260504202
performance on bn (304 examples)
62.82894736842105
performance on fi (314 examples)
43.31210191082803
performance on ja (241 examples)
42.32365145228216
performance on ko (285 examples)
43.15789473684211
performance on ru (237 examples)
54.85232067510548
performance on ar (309 examples)
49.19093851132686
Final macro averaged score: 
49.921052465692526
Evaluating R@5kt
performance on te (238 examples)
63.02521008403361
performance on bn (304 examples)
70.06578947368422
performance on fi (314 examples)
51.910828025477706
performance on ja (241 examples)
53.52697095435685
performance on ko (285 examples)
53.333333333333336
performance on ru (237 examples)
61.60337552742617
performance on ar (309 examples)
57.60517799352751
Final macro averaged score: 
58.72438362740563
```

