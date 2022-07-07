#!/bin/bash

DATA='./data'
mkdir -pv ${DATA}

#Download all dataset from OPUS
echo "Downloading datasets from OPUS:"
declare -a languages=("ja" "ru" "ar" "te" "bn" "fi" "ko")
for lang in ${languages[@]};
do
	if [ ! -d "${DATA}/en-${lang}.WikiMatrix" ]; then
		if [[ ${lang} =~ ^(ar|bn)$ ]]; then
			wget  --output-document ${DATA}/en-${lang}.txt.zip https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/${lang}-en.txt.zip
		else
			wget  --output-document ${DATA}/en-${lang}.txt.zip https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/en-${lang}.txt.zip
		fi
		unzip ${DATA}/en-${lang}.txt.zip -d ${DATA}/en-${lang}.WikiMatrix
		rm ${DATA}/en-${lang}.txt.zip
	fi
done

for lang in bn fi ko;
do
	if [ ! -d "${DATA}/en-${lang}.CCMatrix" ]; then
		if [[ ${lang} =~ ^(bn)$ ]]; then
			wget  --output-document ${DATA}/en-${lang}.txt.zip https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/${lang}-en.txt.zip
		else
			wget  --output-document ${DATA}/en-${lang}.txt.zip https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/en-${lang}.txt.zip
		fi
		unzip ${DATA}/en-${lang}.txt.zip -d ${DATA}/en-${lang}.CCMatrix
		rm ${DATA}/en-${lang}.txt.zip
	fi
done

if [ ! -d "${DATA}/en-te.CCAligned" ]; then
	wget --output-document ${DATA}/en-te.txt.zip https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/en-te.txt.zip
	unzip ${DATA}/en-te.txt.zip -d ${DATA}/en-te.CCAligned
	rm ${DATA}/en-te.txt.zip
fi

# Create tsv file including 7 language for 2 epoch
echo "Creating tsv files:"
source ./script/get_fixed_random.sh

for lang in ${languages[@]};
do
	if [ ! -f "${DATA}/en-${lang}.tsv" ]; then
		if [[ ${lang} =~ ^(ja|ru|ar)$ ]]; then
			paste ${DATA}/en-${lang}.WikiMatrix/WikiMatrix.*.en ${DATA}/en-${lang}.WikiMatrix/WikiMatrix.*.${lang} > ${DATA}/en-${lang}.tsv
		elif [[ ${lang} =~ ^(bn|fi|ko)$ ]]; then
			paste ${DATA}/en-${lang}.WikiMatrix/WikiMatrix.*.en ${DATA}/en-${lang}.WikiMatrix/WikiMatrix.*.${lang} > ${DATA}/en-${lang}.tsv.1
			paste ${DATA}/en-${lang}.CCMatrix/CCMatrix.*.en ${DATA}/en-${lang}.CCMatrix/CCMatrix.*.${lang} > ${DATA}/en-${lang}.tsv.2
			shuf --random-source=<(get_fixed_random 42) ${DATA}/en-${lang}.tsv.2 | head -n 1000000 > ${DATA}/en-${lang}.tsv.2.shuffle.1M
			cat ${DATA}/en-${lang}.tsv.1 ${DATA}/en-${lang}.tsv.2.shuffle.1M > ${DATA}/en-${lang}.tsv
			rm ${DATA}/en-${lang}.tsv.[12]*
		else
			paste ${DATA}/en-${lang}.WikiMatrix/WikiMatrix.*.en ${DATA}/en-${lang}.WikiMatrix/WikiMatrix.*.${lang} > ${DATA}/en-${lang}.tsv.1
			paste ${DATA}/en-${lang}.CCAligned/CCAligned.*.en ${DATA}/en-${lang}.CCAligned/CCAligned.*.${lang} > ${DATA}/en-${lang}.tsv.2
			cat ${DATA}/en-${lang}.tsv.1 ${DATA}/en-${lang}.tsv.2 > ${DATA}/en-${lang}.tsv
			rm ${DATA}/en-${lang}.tsv.[12]*
		fi
	fi
done

if [ ! -f "${DATA}/en-7lan_2ep.tsv" ]; then
	cat ${DATA}/en-[a-z]*.tsv ${DATA}/en-[a-z]*.tsv | shuf --random-source=<(get_fixed_random 42) -o ${DATA}/en-7lan_2ep.tsv
fi

# Create triples
echo "Creating triples and deleting intermediate files:"
if [ ! -f "${DATA}/en-7lan_2ep_triple.en.clean" ]; then
	# Remove badlines
	python ./script/remove_badlines.py --input_file ${DATA}/en-7lan_2ep.tsv --output_file ${DATA}/en-7lan_2ep.tsv.clean
	cut -f 1 ${DATA}/en-7lan_2ep.tsv.clean > ${DATA}/en-7lan_2ep.en
	cut -f 2 ${DATA}/en-7lan_2ep.tsv.clean > ${DATA}/en-7lan_2ep.other
	paste ${DATA}/en-7lan_2ep.en ${DATA}/en-7lan_2ep.en ${DATA}/en-7lan_2ep.en > ${DATA}/en-7lan_2ep_triple.en.clean
	paste ${DATA}/en-7lan_2ep.other ${DATA}/en-7lan_2ep.en ${DATA}/en-7lan_2ep.en > ${DATA}/en-7lan_2ep_triple.other.clean
	rm -rf ${DATA}/en-7lan_2ep.* ${DATA}/en-7lan_2ep_triple.en ${DATA}/en-7lan_2ep_triple.other ${DATA}/en-7lan_2ep.tsv.clean ${DATA}/en-*.tsv ${DATA}/en-[a-z]*
fi

echo "Done"