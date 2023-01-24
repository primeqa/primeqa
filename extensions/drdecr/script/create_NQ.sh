#!/bin/bash

DATA='./data'
mkdir -pv ${DATA}

#Generating NQ training triples needs three files:
#3_20_biased200.json
#psgs_w100.tsv.gz
#train-questions.tsv

echo "Downloading 3_20_biased200.json"
if [ ! -f "${DATA}/3_20_biased200.json" ]; then
	wget --output-document ${DATA}/3_20_biased200.json \
	https://storage.googleapis.com/okhattab/share/OpenQA/2021-Mar/experiments/Feb26.NQ/triples/ColBERT.C3/3_20_biased200.json
fi

echo "Creating psgs_w100.tsv"
if [ ! -f "${DATA}/psgs_w100.tsv" ]; then
	wget --output-document ${DATA}/psgs_w100.tsv.gz \
	https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
	gzip -d ${DATA}/psgs_w100.tsv.gz
fi

echo "Creating train-questions.tsv"
if [ ! -f "${DATA}/train-questions.tsv" ]; then
	wget --output-document ${DATA}/nq-train.qa.csv \
	https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv
	cut -f1 ${DATA}/nq-train.qa.csv | nl -v 0 | sed 's/^ *//g' > ${DATA}/train-questions.tsv
	rm ${DATA}/nq-train.qa.csv
fi

echo "Creating NQ training triples"
if [ ! -f "${DATA}/ColBERT.C3_3_20_biased200_triples_text.tsv" ]; then
	python ./script/convert_triples_to_text.py \
	--queries ${DATA}/train-questions.tsv \
	--triples ${DATA}/3_20_biased200.json \
	--collection ${DATA}/psgs_w100.tsv \
	--out ${DATA}/ColBERT.C3_3_20_biased200_triples_text.tsv
	echo "Removing intermediate files"
	rm ${DATA}/3_20_biased200.json ${DATA}/train-questions.tsv
fi

echo "Done"