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
        # encoding['labels'] = torch.zeros(512)
        # encoding['numeric_values_scale'] = torch.ones(512)
        # encoding['numeric_values']= torch.tensor([float("nan")]*512)
        #print(encoding)
        #print("Float answer",item.float_answer)
        if item.float_answer:
            encoding["float_answer"] = torch.tensor(item.float_answer)
        return encoding
    def __len__(self):
       return len(self.data)

class TableQADataset:
    def __init__(self,dataset_name,tokenizer=None):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        
    def load_data(self,split):
        if self.dataset_name=="sqa":
            if split=="train":
                data = pd.read_csv('primeqa/tableqa/preprocessors/data/sqa_1.0/random-split-1-train.tsv', sep='\t')
                train_dataset = DatasetProcessor(data, self.tokenizer,'primeqa/tableqa/preprocessors/data/sqa_1.0/')
                return train_dataset
            elif split=="dev":
                data = pd.read_csv('primeqa/tableqa/preprocessors/data/sqa_1.0/random-split-1-dev.tsv', sep='\t')
                dev_dataset = DatasetProcessor(data, self.tokenizer,'primeqa/tableqa/preprocessors/data/sqa_1.0/')
                return dev_dataset
            elif split=="test":
                data = pd.read_csv('primeqa/tableqa/preprocessors/data/sqa_1.0/test.tsv', sep='\t')
                test_dataset = DatasetProcessor(data, self.tokenizer,'primeqa/tableqa/preprocessors/data/sqa_1.0/')
                return test_dataset
        elif self.dataset_name=="wikisql":
            if split =="train":
                data = pd.read_csv('primeqa/tableqa/preprocessors/data/wikisql/train.tsv', sep='\t')
                dataset = DatasetProcessor(data, self.tokenizer,'primeqa/tableqa/preprocessors/data/wikisql/')
            elif split=="dev":
                print("here")
                data = pd.read_csv('primeqa/tableqa/preprocessors/data/wikisql/dev.tsv', sep='\t')
                dataset = DatasetProcessor(data, self.tokenizer,'primeqa/tableqa/preprocessors/data/wikisql/')
            return dataset


