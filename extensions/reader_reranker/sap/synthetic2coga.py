# read in the file and then check if can
# find sentence with high rouge to consider as answer passage in the document.
# Finally, convert to SQuAD format.

import pandas as pd
from primeqa.mrc.metrics.rouge.rouge import ROUGE
import re
import json
import glob
import csv

data = []

loio_map_file = "/dccstor/srosent3/reranking/loio_map.tsv"

loio_map = pd.read_csv(loio_map_file, delimiter="\t", names=['document_id', 'loio'])

# load existing questions with docs ids
with open('/dccstor/srosent3/reranking/sap_genq/more_neg/full/synthetic_questions_squadformat_pos.jsonl') as f:
    for line in f:
        json_line = json.loads(line)
        data.append([json_line['id'],json_line['question'],"NA",loio_map[loio_map['document_id']==json_line['document_id']].values[0][1],None,None,None,None,json_line['answers']['text'][0]])

df = pd.DataFrame(data, columns=["Count","Question","passage 1","loio 1","passage 2","loio 2","passage 3","loio 3","Gold answer"])
df.to_csv('/dccstor/srosent3/reranking/sap_genq/more_neg/full/synthetic_questions_pos_coga_format.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)