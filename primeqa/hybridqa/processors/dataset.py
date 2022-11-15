from torch.utils.data import Dataset,DataLoader
import random
import pandas as pd
import torch
from blingfire import text_to_sentences
import numpy as  np

class TableQADataset(Dataset):
    def __init__(self,data,max_seq_length,passage_tokenizer,bert_tokenizer,shuffle,ret_labels=True):
        self.data = data
        self.max_seq_len = max_seq_length
        self.passage_tokenizer = passage_tokenizer
        self.bert_tokenizer = bert_tokenizer
        #self.shuffle = shuffle
        self.ret_labels = ret_labels
        self.row_inputs = []
        labels = []
        passages =[]
        table_row_list = []
        question_list = []
        self.question_ids_list = []
        for d in self.data:
            question_str = d['question']
            question_id = d['question_id']
            #question = [d['question']]
            question_list.append(question_str)
            #q_input = self.tokenizer(question,add_special_tokens=True, truncation=True,padding=True, return_tensors='pt', max_length = self.max_seq_len)
            table_row_list.append(str(d['table_row']))
            #tablepd = pd.DataFrame(d['table_row'],index=[0])
            self.question_ids_list.append(question_id)
            #row_input = self.tapas_tokenizer(table=tablepd, queries=question, padding="max_length", return_tensors="pt")
            passages.append(d['table_passage_row'])
            if self.ret_labels:
                labels.append(d['label'])
            #self.row_inputs.append(row_input)
        #table_pd = pd.DataFrame(table_row_list)
        #print(table_pd)
        self.question_inputs = self.bert_tokenizer(question_list,add_special_tokens=True, truncation=True, padding=True, return_tensors='pt', max_length = 256)

        self.row_inputs = self.bert_tokenizer(table_row_list,add_special_tokens=True, truncation=True, padding=True, return_tensors='pt', max_length = 256)

        self.passage_inputs = self.passage_tokenizer(passages,add_special_tokens=True, truncation=True, padding=True, return_tensors='pt', max_length = self.max_seq_len )
        if self.ret_labels:
            self.labels = torch.tensor(labels)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if self.ret_labels:
            return {key: self.question_inputs[key][index] for key in self.question_inputs.keys()},{key: self.row_inputs[key][index] for key in self.row_inputs.keys()}, {key: self.passage_inputs[key][index] for key in self.passage_inputs.keys()},self.labels[index]
        else:
            return self.question_ids_list[index],{key: self.question_inputs[key][index] for key in self.question_inputs.keys()},{key: self.row_inputs[key][index] for key in self.row_inputs.keys()}, {key: self.passage_inputs[key][index] for key in self.passage_inputs.keys()}



class TableQADatasetQRSconcat(Dataset):
    def __init__(self,data,max_seq_length,bert_tokenizer,use_st_out=False, ret_labels=True):
        self.data = data
        self.max_seq_len = max_seq_length
        self.bert_tokenizer = bert_tokenizer
        #self.shuffle = shuffle
        self.ret_labels = ret_labels
        self.row_inputs = []
        self.sep_token = "[SEP]"
        self.cls_token = "[CLS]"
        self.val_sep = "[unused3]"
        self.q_r_sep = "[unused4]"
        self.col_sep = "[unused5]"
        self.positive_count = 0
        self.negative_count = 0
        self.labels_list = []
        q_r_list =[]
        self.question_ids_list = []
        for d in self.data:
            question_str = d['question']
            question_id = d['question_id']
            table_row = d['table_row']
            #evidence_sent = d['evidence_sent']
            gold_sentences = []
            if use_st_out:
                gold_sentences = d['table_passage_row']
            else:
                if 'st_out' in d:
                    gold_sentences = self.get_sentence_containing_answer_text(d['table_passage_row'],d['answer-text'],d['st_out']) 
                else:
                    gold_sentences = self.get_sentence_containing_answer_text(d['table_passage_row'],d['answer-text'],'') 
            q_r_concat = self.concat_question_table_row(question_str,table_row,gold_sentences)
            self.question_ids_list.append(question_id)
            q_r_list.append(q_r_concat)
            if self.ret_labels:
                self.labels_list.append(d['label'])
                if d['label']==1:
                    self.positive_count+=1
                else:
                    self.negative_count+=1
            #self.row_inputs.append(row_input)
        #table_pd = pd.DataFrame(table_row_list)
        #print(table_pd)
        self.q_r_inputs = self.bert_tokenizer(q_r_list,add_special_tokens=False, truncation=True, padding=True, return_tensors='pt', max_length = 512)


        if self.ret_labels:
            self.labels = torch.tensor(self.labels_list)

    def concat_question_table_row(self,question_str,table_row,gold_sentences):
        question_str = self.cls_token+" "+question_str+" "+self.sep_token+" "
        table_str = ""
        for c,r in table_row.items():
            table_str+=str(c)+" is "+str(r)+" . " # table_str+=str(c)+" "+self.val_sep+" "+str(r)+" "+self.col_sep+" "
        question_str = question_str+table_str+" "+self.sep_token+" "+gold_sentences
        #print("question Annotated",question_str)
        return question_str
    def get_sentence_containing_answer_text(self,table_passage_row,answer_text,st_out_text):
        sentences = []
        all_sentences = text_to_sentences(table_passage_row).split("\n")
        for s in all_sentences:
            if answer_text.lower() in s.lower():
                sentences.append(s)
        # adding sentences filtered using sentence-transformer similarity measure.
        sentences.extend(text_to_sentences(st_out_text).split("\n")) 
        # permuting sentences.
        sentences = np.random.permutation(sentences)
        sentences = ' '.join(sentences)
        return sentences

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if self.ret_labels:
            return {key: self.q_r_inputs[key][index] for key in self.q_r_inputs.keys()},self.labels[index]
        else:
            return self.question_ids_list[index],{key: self.q_r_inputs[key][index] for key in self.q_r_inputs.keys()}

class TableQADatasetQRconcat(Dataset):
    def __init__(self,data,max_seq_length,bert_tokenizer,shuffle,ret_labels=True):
        self.data = data
        self.max_seq_len = max_seq_length
        self.bert_tokenizer = bert_tokenizer
        #self.shuffle = shuffle
        self.ret_labels = ret_labels
        self.row_inputs = []
        self.sep_token = "[SEP]"
        self.cls_token = "[CLS]"
        self.val_sep = "[unused3]"
        self.q_r_sep = "[unused4]"
        self.col_sep = "[unused5]"
        self.positive_count = 0
        self.negative_count = 0
        self.labels_list = []
        q_r_list =[]
        self.question_ids_list = []
        for d in self.data:
            question_str = d['question']
            question_id = d['question_id']
            table_row = d['table_row']

            q_r_concat = self.concat_question_table_row(question_str,table_row)
            self.question_ids_list.append(question_id)
            q_r_list.append(q_r_concat)
            if self.ret_labels:
                self.labels_list.append(d['label'])
                if d['label']==1:
                    self.positive_count+=1
                else:
                    self.negative_count+=1
            #self.row_inputs.append(row_input)
        #table_pd = pd.DataFrame(table_row_list)
        #print(table_pd)
        self.q_r_inputs = self.bert_tokenizer(q_r_list,add_special_tokens=False, truncation=True, padding=True, return_tensors='pt', max_length = 512)


        if self.ret_labels:
            self.labels = torch.tensor(self.labels_list)

    def concat_question_table_row(self,question_str,table_row):
        question_str = self.cls_token+" "+question_str+" "+self.q_r_sep+" "
        table_str = ""
        for c,r in table_row.items():
            table_str+=str(c)+" "+self.val_sep+" "+str(r)+" "+self.col_sep+" "
        question_str = question_str+table_str
        #print(question_str)
        return question_str

    def get_sentence_containing_answer_text(self,table_passage_row,answer_text):
        sentences = []
        all_sentences = text_to_sentences(table_passage_row).split("\n")
        for s in all_sentences:
            if answer_text.lower() in s.lower():
                sentences.append(s)
        #print(sentences)
        sentences = (self.sep_token).join(sentences)
         
        return sentences

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if self.ret_labels:
            return {key: self.q_r_inputs[key][index] for key in self.q_r_inputs.keys()},self.labels[index]
        else:
            return self.question_ids_list[index],{key: self.q_r_inputs[key][index] for key in self.q_r_inputs.keys()}
