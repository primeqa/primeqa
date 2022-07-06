from cmath import nan
import torch
import pandas as pd
from nlp import load_dataset
import nlp
import ast

class DatasetProcessor(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer,table_csv_path):
        self.data = data
        self.tokenizer = tokenizer
        self.table_csv_path = table_csv_path
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        table = pd.read_csv(self.table_csv_path + item.table_file,index_col=0).astype(str) # be sure to make your table data text only
        answer_coordinates=ast.literal_eval(str(item.answer_coordinates))

        encoding = self.tokenizer(table=table,
                                queries=item.question,
                                answer_coordinates=answer_coordinates,
                                answer_text=item.answer_text,
                                truncation=True,
                                padding="max_length",
                                return_tensors="pt",
                                max_column_id=32,
                                max_row_id=64,
        )
     
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        if item.float_answer:
            encoding["float_answer"] = torch.tensor(item.float_answer)
        return encoding
    def __len__(self):
       return len(self.data)


class TableQADataset:
    def __init__(self,data_path_root,train_dataset_path,dev_dataset_path,tokenizer=None):
        self.tokenizer = tokenizer
        self.data_path_root = data_path_root
        self.train_dataset_path = train_dataset_path
        self.dev_dataset_path = dev_dataset_path
        
    def load_data(self):
        train_data = pd.read_csv(self.train_dataset_path, sep='\t')
        train_dataset = DatasetProcessor(train_data, self.tokenizer,self.data_path_root)

        dev_data = pd.read_csv(self.dev_dataset_path, sep='\t')
        dev_dataset = DatasetProcessor(dev_data, self.tokenizer,self.data_path_root)
        return train_dataset,dev_dataset



