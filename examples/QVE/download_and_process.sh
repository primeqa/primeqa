#! /bin/bash

cd data

declare -a arr=("SQuAD" "NewsQA" "NaturalQuestionsShort" "HotpotQA" "TriviaQA-web")

##downloading the datasets from the MRQA 2019.
##We use the dev set as the test set

for dataset_name in "${arr[@]}"; do
  echo "Downloading dataset: $dataset_name ..."
  wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/"$dataset_name".jsonl.gz -O "$dataset_name".train.jsonl.gz
  wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/"$dataset_name".jsonl.gz -O "$dataset_name".test.jsonl.gz

  ##convert mrqa dataset to primeqa data format file
  python convert_mrqa_to_custom_data_format.py \
  --input_file "$dataset_name".train.jsonl.gz \
  --output_file "$dataset_name".train.jsonl

  python convert_mrqa_to_custom_data_format.py \
  --input_file "$dataset_name".test.jsonl.gz \
  --output_file "$dataset_name".test.jsonl

  rm "$dataset_name".train.jsonl.gz
  rm "$dataset_name".test.jsonl.gz

  ##For all the target domain datasets {"NewsQA" "NaturalQuestionsShort" "HotpotQA" "TriviaQA-web"}
  ##We sample 1000 QAs from the training as the dev set
  if [ "$dataset_name" != "SQuAD" ]; then
    echo "Sampling dev set from the training set..."
    python split_data_num.py \
    --in_file "$dataset_name".train.jsonl \
    --out_file_dev "$dataset_name".sample.dev.jsonl \
    --out_file_train "$dataset_name".sample.train.jsonl \
    --num 1000
  fi
done
