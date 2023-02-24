from nltk.tokenize import word_tokenize, sent_tokenize
import urllib.parse
import sys
import json
import gzip

def tokenize(string, remmove_dot=False):
    def func(string):
        return " ".join(word_tokenize(string))
    
    string = string.rstrip('.')
    return func(string)

def url2dockey(string):
    string = urllib.parse.unquote(string)
    return string

def filter_firstKsents(string, k):
    string = sent_tokenize(string)
    string = string[:k]
    return " ".join(string)

def compressGZip(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)

    json_str = json.dumps(data) + "\n"               # 2. string (i.e. JSON)
    json_bytes = json_str.encode('utf-8')            # 3. bytes (i.e. UTF-8)

    with gzip.GzipFile(file_name + '.gz', 'w') as fout:   # 4. gzip
        fout.write(json_bytes)

def readGZip(file_name):
    if file_name.endswith('gz'):
        with gzip.GzipFile(file_name, 'r') as fin:    # 4. gzip
            json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)

        json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
        data = json.loads(json_str)                      # 1. data
        return data
    else:
        with open(file_name, 'r') as fin:
            data = json.load(fin)
        return data