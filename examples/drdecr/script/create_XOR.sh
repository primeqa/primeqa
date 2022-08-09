#!/bin/bash

DATA='./data/XOR'
mkdir -pv ${DATA}

echo "Downloading the DPR corpus of English Wikpedia"
if [ ! -f "${DATA}/psgs_w100.tsv.gz" ]; then
	wget --output-document ${DATA}/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
fi

echo "Formating into JSON"
if [ ! -f "${DATA}/psgs_w100_pyserini.jsonl" ]; then
	python ../../primeqa/ir/scripts/xortydi/convert_corpus_tsv_to_pyserini_jsonl.py \
	--input ${DATA}/psgs_w100.tsv.gz \
	--output ${DATA}
fi

echo "Building the Pyserini index"
if [ ! -d "${DATA}/index" ]; then
	python -m pyserini.index -collection JsonCollection \
	-input ${DATA}/ \
	-generator DefaultLuceneDocumentGenerator \
	-threads 1 \
	-storePositions \
	-storeDocvectors \
	-storeRaw \
	-index ${DATA}/index
fi

echo "Generating additional training examples"
mkdir -pv ${DATA}/question_translation/
declare -a languages=("ja" "ru" "ar" "te" "bn" "fi" "ko")
for lang in ${languages[@]};
do
	if [ ! -d "${DATA}/${lang}-en" ]; then
		wget  --output-document ${DATA}/question_translation/${lang}-en.zip \
		https://nlp.cs.washington.edu/xorqa/XORQA_site/data/${lang}-en.zip
		unzip ${DATA}/question_translation/${lang}-en.zip -d ${DATA}/question_translation/
		rm ${DATA}/question_translation/${lang}-en.zip
	fi
done

echo "Creating triples files"
python ../../primeqa/ir/scripts/xortydi/generate_xorqa_examples.py \
--input_file ${DATA}/dpr_train_data.json \
--index_path ${DATA}/index \
--output_dir ${DATA} \
--question_translations_dir ${DATA}/question_translation/ \
--num_rounds 5 \
--randomize

echo "Remove intermediate files"
mv ${DATA}/xor* ${DATA}/..
rm -rf ${DATA}

echo "Done"