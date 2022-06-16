## Generating BM25 retrieval based training examples for XORTyDI

### Create an Pyserini index of Wikipedia passages for XORTyDI

1. Download the DPR corpus of English Wikpedia (December 20, 2018 dump) split 100 word passages 
   wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
2. Format as JSON
    python convert_corpus_tsv_to_pyserini_jsonl.py --input <psgs_w100_file> --output <output_dir>
3. Build the Pyserini index
   ```
   python -m pyserini.index.lucene --collection JsonCollection --input <psgs_w100_jsonl-dir> --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw --index <index-dir>
   ```
### Generate additional training examples
The script [primeqa/ir/scripts/xortydi/generate_xorqa_examples.py] will generate additional positive and negative examples for XORTyDI using BM25 search over a Wikipedia psssage index.

Positive passages are those containing the answer text and negative passages are those where there 
was no string matching the answer text. 

The following are steps to create the training data:

1. Download training data json from here: https://drive.google.com/file/d/1tB0ehH0Q_V7gEgc2ZoxSlXpHTzUONNNW/view
2. Download English translations of the questions from here and unzip into under one folder <question_translations_dir>.  These translations are used to obtain the English translation of the question which is use to query the English wikipedia index: 
   - https://nlp.cs.washington.edu/xorqa/XORQA_site/data/ar-en.zip
   - https://nlp.cs.washington.edu/xorqa/XORQA_site/data/bn-en.zip
   - https://nlp.cs.washington.edu/xorqa/XORQA_site/data/fi-en.zip
   - https://nlp.cs.washington.edu/xorqa/XORQA_site/data/ja-en.zip
   - https://nlp.cs.washington.edu/xorqa/XORQA_site/data/ko-en.zip
   - https://nlp.cs.washington.edu/xorqa/XORQA_site/data/ru-en.zip
   - https://nlp.cs.washington.edu/xorqa/XORQA_site/data/te-en.zip
3. Run command:
```shell
   python primeqa/ir/scripts/xortydi/generate_xorqa_examples.py --input_file <training-data-json>
   --index_path <wiki_psgs_w100_index_path> --question_translations_dir <question_translation_dir>
   --num_rounds 5 --randomize 
```
4. This will create 3 files in the output directory:
   - ```xortydi_ir_negs_poss.json``` contains elements ```"ir_negative_ctxs"``` and  ```"ir_positive_ctxs"```
   - ```xorqa_triples_3poss_100neg_5ep_randTrue.tsv``` (17125570 lines) contains training data formatted for ColBERT. Here the question is in the original language
   - ```xorqa_triples_3poss_100neg_en_5ep_randTrue.tsv``` (17125570 lines) contains training data Formatted for ColBERT. Here the question has been translated to English


