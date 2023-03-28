from datasets.arrow_dataset import Batch
from transformers import BatchEncoding
from datasets import Dataset
from typing import Tuple, List
import torch

from primeqa.mrc.processors.preprocessors.abstract import AbstractPreProcessor

class ELI5FiDPreprocessor(AbstractPreProcessor):
    
    _ignore_pad_token_for_loss = True
    
    _question_column = "input"
    
    _answer_column = "output"
    
    _context_column = "passages"
    
    def adapt_dataset(self, dataset: Dataset, is_train: bool) -> Dataset:
        pass
    
    def label_features_for_subsampling(self, tokenized_examples: BatchEncoding, examples: Batch) -> BatchEncoding:
        pass
    
    def subsample_features(self, dataset: Dataset) -> Dataset:
        pass
    
    def validate_schema(self, dataset: Dataset, is_train: bool, pre_adaptation: bool = True) -> None:
        pass
    
    def process_train(self, examples: Dataset) -> Tuple[Dataset, Dataset]:
        return self._process(examples, is_train=True)

    def process_eval(self, examples: Dataset) -> Tuple[Dataset, Dataset]:
        return self._process(examples, is_train=False)
    
    def _process(self, examples: Dataset, is_train: bool) -> Tuple[Dataset, Dataset]:
         mode = "train" if is_train else "eval"
         dataset = examples.map(
                self.preprocess_eli5_function_fid,
                fn_kwargs=dict(mode=mode),
                batched=True,
                num_proc=self._num_workers,
                remove_columns=examples.column_names,
                load_from_cache_file=self._load_from_cache_file,
                desc=f"Running tokenizer on {mode} dataset",
            )
         return examples, dataset
    
    def preprocess_eli5_function_fid(self, examples: Dataset, mode: str) -> Dataset:
        indexes, inputs, targets = self.preprocess_eli5_batch_fid(examples, mode=mode)
        passage_ids, passage_masks = self.encode_passages(inputs)
        #TODO:  padding is set to True, should be in the input args
        padding = "max_length"
        if targets:
            with self._tokenizer.as_target_tokenizer():
                labels = self._tokenizer(targets, max_length=self._max_answer_len, padding=padding, truncation=True)
            if padding == "max_length" and self._ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self._tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
        model_inputs = {}
        model_inputs["input_ids"] = passage_ids
        model_inputs["attention_mask"] = passage_masks
        if targets:
            model_inputs["labels"] = labels["input_ids"]
        model_inputs["example_id"] = indexes
        return model_inputs
    
    def encode_passages(self,batch_text_passages):
        '''
        Param: 
            batch_text_passages: (bsz, n_doc, )
            all passages are encoded and padded to max_length
            not using max padding will complicate the FID Data Collator
            the input in the FID system does not need to be padded again
        '''
        passage_ids, passage_masks = [], []
        for text_passages in batch_text_passages:
            p = self._tokenizer(
                text_passages,
                padding='max_length',
                max_length=self._max_seq_len,
                return_tensors='pt',
                truncation=True
            )
            passage_ids.append(p['input_ids'][None])
            passage_masks.append(p['attention_mask'][None])

        passage_ids = torch.cat(passage_ids, dim=0)
        passage_masks = torch.cat(passage_masks, dim=0)
        return passage_ids.tolist(), passage_masks.tolist()
    
    
    def preprocess_eli5_batch_fid(self, examples, mode="train") -> Tuple[List[str], List[str]]:
        indices = []
        questions = examples[self._question_column]
        contexts = examples[self._context_column]
        n_doc = self._max_contexts

        def top_passages(ctx):
            assert n_doc <= len(ctx) 
            return [ctx[i]["text"] for i in range(n_doc)]
        def append_question(passages, question):
            return [f"question: {question} passage: {t}" for t in passages]
        # multiple answers for training
        if mode == "train":
            answers = examples[self._answer_column]
            inputs = []
            targets = []
            for idx,q in enumerate(questions):
                if len(q) == 0: 
                    # Skip empty questions
                    continue
                passages = top_passages(contexts[idx])
                question_passages = append_question(passages, q)
                answer_list = answers[idx]
                if len(answer_list) == 0:
                    inputs.append(question_passages)
                    targets.append("")  
                    indices.append(examples["id"][idx])
                else: # multiple answers
                    for answer_data in answer_list:
                        a = answer_data["answer"]
                        answer_score = answer_data["meta"]["score"]     
                        if answer_score >= 3: # only takes answers whose score>3
                            inputs.append(question_passages)
                            targets.append(a)
                            indices.append(examples["id"][idx])
                        
        elif mode == "eval": # for evaluation only take each question once
            inputs = []
            if self._answer_column in examples:
                answers = examples[self._answer_column]
            else:
                answers = []
            for idx,q in enumerate(questions):
                passages = top_passages(contexts[idx])
                question_passages = append_question(passages, q)
                inputs.append(question_passages)
                indices.append(examples["id"][idx])
            targets = [answer[0]["answer"] if len(answer) > 0 else "" for answer in answers]
        else:
            raise ValueError("mode requires eval or train")

        return indices, inputs, targets # inputs is a list of a list of question+passage, targets is a list of answers

    def set_max_contexts(self, new_max_contexts):
        self._max_contexts = new_max_contexts