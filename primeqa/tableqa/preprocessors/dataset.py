import torch
import pandas as pd

#tsv_path = "your_path_to_the_tsv_file"
#table_csv_path = "your_path_to_a_directory_containing_all_csv_files"

class TableDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        item = data.iloc[idx]
        table = pd.read_csv(table_csv_path + item.table_file).astype(str) # be sure to make your table data text only
        encoding = self.tokenizer(table=table,
                                  queries=item.question,
                                  answer_coordinates=item.answer_coordinates,
                                  answer_text=item.answer_text,
                                  truncation=True,
                                  padding="max_length",
                                  return_tensors="pt"
        )
        # remove the batch dimension which the tokenizer adds by default
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        # add the float_answer which is also required (weak supervision for aggregation case)
        encoding["float_answer"] = torch.tensor(item.float_answer)
        return encoding
    def __len__(self):
       return len(self.data)

#data = pd.read_csv(tsv_path, sep='\t')
#train_dataset = TableDataset(data, tokenizer)
#train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)