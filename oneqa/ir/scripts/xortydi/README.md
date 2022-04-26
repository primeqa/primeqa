## BM25 retrieval


Currently supports Pyserini indexing and search.

The instructions below are for creating an index of English Wikipedia passage and use these to search 
and evaluate performance on the Google translation of the XORTyDI DEV set queries. 

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






