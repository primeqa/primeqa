#! /bin/bash

dataset_name=TriviaQA-web
SOURCE_QG_CKPT_DIR=checkpoints/QG_source_only
TARGET_DEV_QG_CKPT_DIR=checkpoints/QG_source_then_"$dataset_name"_dev

mkdir checkpoints

if ! [ -d "$SOURCE_QG_CKPT_DIR" ]; then


  echo "##########################################################################"
  echo "Pretrain QG model on SQuAD"
  echo "##########################################################################"

  python primeqa/qg/run_qg.py\
    --model_name_or_path facebook/bart-base \
    --modality passage \
    --train_file data/SQuAD.train.jsonl \
    --eval_file data/SQuAD.test.jsonl \
    --do_train \
    --do_eval \
    --output_dir $SOURCE_QG_CKPT_DIR \
    --learning_rate 3e-5 \
    --num_train_epochs 3

fi


if [ -d "$TARGET_DEV_QG_CKPT_DIR" ]; then
  echo "##########################################################################"
  echo "$TARGET_DEV_QG_CKPT_DIR exists. Skip Training."
  echo "##########################################################################"
else
  echo "##########################################################################"
  echo "Finetuning source QG on the target ($dataset_name) dev set"
  echo "##########################################################################"

  mkdir data/"$dataset_name"_QG/

    python primeqa/qg/run_qg.py\
    --model_name_or_path $SOURCE_QG_CKPT_DIR \
    --modality passage \
    --train_file data/"$dataset_name".sample.dev.jsonl \
    --eval_file data/"$dataset_name".test.jsonl \
    --do_train \
    --do_eval \
    --output_dir $TARGET_DEV_QG_CKPT_DIR \
    --learning_rate 3e-5 \
    --num_train_epochs 3
fi


echo "##########################################################################"
echo "Generate synthetic QAs on $dataset_name training contexts based on $TARGET_DEV_QG_CKPT_DIR"
echo "##########################################################################"


 python primeqa/examples/QVE/run_qq_inference.py \
 --model_name $TARGET_DEV_QG_CKPT_DIR \
 --input_file data/"$dataset_name".sample.train.jsonl \
 --output_file data/"$dataset_name"_QG/"$dataset_name".train.targetfinedtuned.gen.jsonl \








