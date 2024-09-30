import typing
import os
import ujson as json
import csv
from pathlib import Path
from primeqa.util.file_utils import read_open
from tqdm import tqdm
from tqdm import tqdm


def lookup_by_aliases(jobj: dict, field_options: typing.List[str], *, default):
    for f in field_options:
        if f in jobj:
            return jobj[f]
    return default


class Passage:
    __slots__ = 'pid', 'title', 'text'

    def __init__(self, pid: str, title: str, text: str):
        self.pid = pid
        self.title = title
        self.text = text

    def to_dict(self):
        """
        Useful for json serialization
        :return: dictionary representation of the passage
        """
        return {slot: getattr(self, slot) for slot in self.__slots__}

    @staticmethod
    def from_dict(jobj: dict):
        pid = lookup_by_aliases(jobj, ['pid', 'id'], default='')
        title = lookup_by_aliases(jobj, ['title'], default='')
        text = lookup_by_aliases(jobj, ['text', 'contents'], default='')
        return Passage(pid, title, text)


def is_tsv(filename: str):
    return any(filename.endswith(ext) for ext in
               ['.tsv', '.tsv.gz', 'tsv.bz2'])
               # '.csv', '.csv.gz', '.csv.bz2'
               
def is_csv(filename: str):
    return any(filename.endswith(ext) for ext in
               ['.csv', '.csv.gz', '.csv.bz2'])

def is_csv(filename: str):
    return any(filename.endswith(ext) for ext in
               ['.csv', '.csv.gz', '.csv.bz2'])
    
def list_corpus_files(*input_files: typing.Union[str, bytes, os.PathLike]):
    all_files = []
    for input_file in input_files:
        input_file = str(input_file)
        if not os.path.exists(input_file):
            raise ValueError(f'No such file: {input_file}')
        if os.path.isdir(input_file):
            sub_files = [str(f) for f in Path(input_file).glob("**/*")]
            all_files.extend([f for f in sub_files if not os.path.isdir(f)])
        else:
            all_files.append(input_file)
    all_files.sort()
    return all_files


# CONSIDER: make this a class with an __iter__ method that returns this generator
def corpus_reader(*input_files: typing.Union[str, bytes, os.PathLike], fieldnames=None):
    for file in list_corpus_files(*input_files):
        with read_open(file) as f:
            if is_tsv(file) or is_csv(file):
                delimiter = ',' if is_csv(file) else '\t'
                reader = csv.DictReader(f, delimiter=delimiter, fieldnames=fieldnames)
                for row in reader:
                    passage = Passage.from_dict(row)
                    yield passage
            else:
                for line in f:
                    jobj = json.loads(line)
                    passage = Passage.from_dict(jobj)
                    
class DocumentCollection:  

    def __init__(self, input_files: typing.Union[str, bytes, os.PathLike], fieldnames=None):
        """ This class provides helper functions to load in a corpus tsv, csv or json file where each row is a 
        document/passage and optionally has a documnet title and id.  If there is no id provided one will 
        assigned starting at 1. 
        
        Args: 
            input_files: list[str] one or more input files
            
        """
        self.reader = corpus_reader(input_files, fieldnames=fieldnames)
        self.id_to_document = None
        self.load_corpus()
        print(len(self.id_to_document))
    
    
    def load_corpus(self):
        """
           Load the corpus tsv/csv or json
        """
        num_docs = 0
        self.id_to_document = {}
        for passage in tqdm(self.reader):
            document = {
                'text': passage.text,
                'title': passage.title,
                "id": str(num_docs + 1) if len(passage.pid) == 0 else passage.pid
            }
            self.id_to_document[document['id']] = document
            num_docs += 1

                
    def write_corpus_tsv(self, output_file: str):
        """
            Write out the corpus in a format ready for indexing. 

        Args:
            output_file (str): tsv file where each row is in format 'id\ttext\title'
        """
        if self.id_to_document == None:
            self.load_corpus()
            
        with open(output_file,'w') as f:
            fieldnames = ['id', 'text', 'title']
            tsv_writer = csv.DictWriter(f, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
            tsv_writer.writeheader()
            tsv_writer.writerows(self.id_to_document.values())
    
    def add_document_text_to_hit(self, hits: list):
        """
        Look up and add document text/title to the hits

        Args:
            hits: list of (document_id, score) tuples

        Returns:
            list[dict]: list of dict 
            {
                'document': document_dict,
                'score': score
            }
        """
        search_results_with_docs = []
        for hit in hits:
            search_results_with_docs.append(
                {
                    'document': self.id_to_document[hit[0]],
                    'score': hit[1]
                }
            )
        
        return search_results_with_docs
        
    
    

    
    
        
        
   
        
        
        
        
    
