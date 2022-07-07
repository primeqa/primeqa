import typing
import os
import ujson as json
import csv
from pathlib import Path
from primeqa.util.file_utils import read_open


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
            if is_tsv(file):
                reader = csv.DictReader(f, delimiter='\t', fieldnames=fieldnames)
                for row in reader:
                    passage = Passage.from_dict(row)
                    yield passage
            else:
                for line in f:
                    jobj = json.loads(line)
                    passage = Passage.from_dict(jobj)
                    yield passage
