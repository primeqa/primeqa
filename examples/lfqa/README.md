
# Long Form Question Answering (LFQA)

LFQA is a form of generative question answering. Given a question,  the retriever component retrieves the supporting documents and a reader component generates the answer conditioned on the supporting documents.  The system generates complex multi-sentence answers. 

## KILT-ELI5

The following shows how to build retriever and reader components to generate answers for the KILT-ELI5 dataset.

Before continuing below make sure you have PrimeQA [installed](https://primeqa.github.io/primeqa/installation.html).

### 1. Download the data from [facebookresearch/KILT](https://github.com/facebookresearch/KILT) 


 Store the data into the `$KILT_ELI5/data` directory.

The KILT knowledge source (the corpus) can be downloaded here: [kilt_knowledgesource.json](http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json) (34.76GiB).<br>
It is based on the [2019/08/01 Wikipedia dump](http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2).<br>

The train set of KILT-ELI5 can be downloaded from [eli5-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/eli5-train-kilt.jsonl) (272,634 lines, 548MiB) 

The dev set of KILT-ELI5 can be downloaded from [eli5-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/eli5-dev-kilt.jsonl) (1,507 lines, 14.1MiB) 

### 2. Preprocess the KILT Knowledge Source

The KILT Knowledge source needs to be preprocessed into a tsv format that can be used by the retrieval component.

```
python examples/lfqa/kilt_passage_corpus.py \
    --kilt_corpus $KILT_ELI5/data/kilt_knowledgesource.json \
    --output_dir $KILT_ELI5/passages/ 
    --passage_ids $KILT_ELI5/kilt_passage_ids.txt
```

### 3. Create a Dense Index with ColBERT

Instructions for training a ColBERT model can be found [here](https://github.com/primeqa/primeqa/blob/main/primeqa/ir/README.md)
We provide a ColBERT model that has already been trained.

```bash
cd $KILT_ELI5
git clone https://huggingface.co/PrimeQA/open-nq-colbert-xlmr-large
COLBERT_CHECKPOINT=$KILT_ELI5/open-nq-colbert-xlmr-large
```

```
python primeqa/ir/run_ir.py \
    --do_index \
    --engine_type ColBERT \
    --amp \
    --doc_maxlen 180 \
    --bsize 256 \
    --model_name_or_path $COLBERT_CHECKPOINT \
    --collection $KILT_ELI5/passages/kilt_knowledgesource_0.tsv \
    --root $KILT_ELI5/colbert_ir \
    --index_name kilt_wikipedia_indname \
    --experiment kilt_wikipedia_exp \
    --nbits 4 \
    --kmeans_niters 10 \
    --num_partitions_max 50000 \
```

### 4. Create the Query Files for the KILT-ELI5 Dataset

The query files are used by the retriever to search the index for supporting documents. Since the KILT-ELI5 dataset has a large number of examples, we recommend creating multiple query files that can be used to perform the search in parallel. 

```
python examples/lfqa/create_ir_queries_from_dataset.py \
    --train_file $KILT_ELI5/data/eli5-train-kilt.jsonl \
    --eval_file $KILT_ELI5/data/eli5-dev-kilt.jsonl \
    --queries_per_file 50000 \
    --output_dir $KILT_ELI5/queries \
```

### 4. Run the Search for the KILT-ELI5 Dataset

The search is designed to run in parallel for chunks of max 50000 queries. 

```
mkdir -p $(dirname $KILT_ELI5/search_results)

query_dir=$KILT_ELI5/queries

query_files=$(for f in ${query_dir}* ; do basename $f ; done)

for EXPT in ${query_files[@]}  ; do \
" \
python primeqa/ir/run_ir.py \
    --do_search \
    --engine_type ColBERT \
    --amp \
    --doc_maxlen 180 \
    --bsize 256 \
    --retrieve_only \
    --queries $KILT_ELI5/queries/${EXPT} \
    --collection $KILT_ELI5/passages/kilt_knowledgesource_0.tsv \
    --model_name_or_path $COLBERT_CHECKPOINT \
    --index_location $KILT_ELI5/colbert_ir/kilt_wikipedia_exp/indexes/kilt_wikipedia_indname 
    --output_dir $KILT_ELI5/search_results/${EXPT}
    --top_k 100 \
    --ncells 4 \
    --centroid_score_threshold 0.4 \
    --ndocs 40000 \
" \
; done
```

### 5. Add the Supporting Passages for the KILT-ELI5 Dataset

```
python examples/lfqa/add_passages_to_dataset.py \
    --train_file $KILT_ELI5/data/eli5-train-kilt.jsonl \
    --eval_file $KILT_ELI5/data/eli5-dev-kilt.jsonl \
    --output_dir $KILT_ELI5/kilt-eli5-colbert-passages \
    --search_result_location $KILT_ELI5/search_results \
    --collection $KILT_ELI5/passages/kilt_knowledgesource_0.tsv \
```

### 6. Run the Reader Component on the KILT-ELI5 Dataset with Supporting Passages

The command below trains and evaluates an FiD generative reader model based on `facebook/bart-large`.<br>
The number of supporting passages is 3. <br>
The max length of the input text is 256. <br>
The max length of the generated answer is 256. <br>
We train the model for 3 epochs and we save the best model at the end. 

```
python primeqa/primeqa/mrc/run_mrc.py \
    --model_name_or_path facebook/bart-large \
    --train_file $KILT_ELI5/kilt-eli5-colbert-passages/train.json \
    --eval_file $KILT_ELI5/kilt-eli5-colbert-passages/dev.json \
    --preprocessor primeqa.mrc.processors.preprocessors.eli5_fid.ELI5FiDPreprocessor \
    --postprocessor primeqa.mrc.processors.postprocessors.eli5_fid.ELI5FiDPostProcessor \
    --task_head primeqa.mrc.models.heads.generative.FID_HEAD \
    --task_model primeqa.mrc.models.fid_task_model.FiDModelForDownstreamTasks \
    --task_data_collator primeqa.mrc.data_models.data_collator.FiDDataCollator \
    --task_trainer primeqa.mrc.trainers.seq2seq_mrc.MRCSeq2SeqTrainer \
    --max_contexts 3 \
    --eval_metrics "ROUGE" \
    --max_seq_length 256 \
    --generation_num_beams 1 \
    --generation_max_length 256 \
    --max_answer_length 256 \
    --predict_with_generate \
    --output_dir KILT_ELI5/eli5_reader_bart_large_fid_colbert \
    --do_train --learning_rate 3e-5 --num_train_epochs 3 \
    --do_eval --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model rougeL \
    --overwrite_output_dir \
    --overwrite_cache \
```
