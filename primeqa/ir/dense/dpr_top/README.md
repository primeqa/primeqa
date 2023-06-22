# DPR with Single Encoder

This is using a single encoder class for both the question and context encoder. 

## Train Script

```
jbsub -q x86_24h -cores 1+1 -mem 100g -require a100_80gb -proj new_dpr -name dpr_train \
python primeqa/primeqa/ir/run_ir.py \
    --engine_type DPR \
    --do_train \
    --train_dir /dccstor/colbert-ir/franzm/data/dpr/biencoder-nq-train.json.gz \
    --output_dir model_default_v2 \
    --num_train_epochs 3 \
    --sample_negative_from_top_k 100 \
    --encoder_gpu_train_limit 32 \
    --max_negatives 1 \
    --max_hard_negatives 1 \
    --bsize 128 \
    --max_grad_norm 1.0 \
    --learning_rate 1e-5 \
    --training_data_type dpr \
    --n_gpu 1 \
    --warmup_fraction 0.1 \
```

## Index Script

```
for EXPT in model_v2_index; do \
for PART in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 ; do \
jbsub -q x86_24h -cores 1+1 -mem 100g -require v100 -proj ${EXPT}_${PART} -name dpr_index \
" \
python primeqa/primeqa/ir/run_ir.py \
    --engine_type DPR \
    --do_index \
    --ctx_encoder_name_or_path /dccstor/mabornea2/oneqa_os/primeqa_dpr/model_default_v2/ctx_encoder \
    --embed ${PART}of16 \
    --sharded_index \
    --collection /dccstor/avi7/neural_ir/colbert/data/psgs_w100.tsv \
    --output_dir ${EXPT} \
    --batch_size 16 \
> log/${EXPT}_${PART} \
2>&1 " ; done ; done
```

## Search Script

```
for EXPT in model_v2_index ; do \
jbsub -q x86_12h -cores 1+1 -mem 100g -require v100 -proj ${EXPT} -name dpr_search \
python primeqa/primeqa/ir/run_ir.py \
    --engine_type DPR \
    --do_search \
    --queries /dccstor/mabornea1/openqa_nq/opennq-queries/dev_0 \
    --index_location /dccstor/mabornea2/oneqa_os/primeqa_dpr/model_v2_index \
    --qry_encoder_name_or_path /dccstor/mabornea2/oneqa_os/primeqa_dpr/model_default_v2/ctx_encoder \
    --model_name_or_path /dccstor/mabornea2/oneqa_os/primeqa_dpr/model_default_v2/ctx_encoder \
    --bsize 16 \
    --output exp_v2_nq_search \
    --top_k 100 \
    --n_gpu 1 \
> log/model_v2_search \
2>&1  ; done
```

## Convert script to start rank 1 and evaluate

The DPR searh results assigns rank 0 for the first passage. Colbert and the evaluation script assumes the rank of the first passage starts from 0. 

```
cat exp_v2_nq_search/ranked_passages.tsv | perl -ane ' $F[2]++; $o = join("\t", @F); print "$o\n";' > exp_v2_nq_search/fixed_ranked_passages.tsv
```

The evaluation script. The results are in the `fixed_ranked_passages.tsv.annotated.metrics` inside the search_directory.

```
python primeqa/ir/dense/colbert_top/utility/evaluate/annotate_EM.py \
--collection /dccstor/avi7/neural_ir/colbert/data/psgs_w100.tsv \
--qas /dccstor/mabornea1/openqa_nq/nq-dev.qa.json \
--ranking /dccstor/mabornea2/oneqa_os/primeqa_dpr/test_nq_search/fixed_ranked_passages.tsv
```


