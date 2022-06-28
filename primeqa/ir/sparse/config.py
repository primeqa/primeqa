from dataclasses import dataclass, field
import dataclasses

@dataclass
class IndexingArguments():

    index_path: str = field(default=None, metadata={"help":"Path to the index directory location"})

    overwrite: bool = field(default=False, metadata={"help": "Overwrite existing directory"})

    corpus_path: str = field(default=None, metadata={"help":"Path to a corpus tsv or json file or directory"})

    fieldnames: list = field(default=None, metadata={"help":"fields names to use to identify document_id, title, text if corpus tsv has no headings"})

    additional_indexing_args: str = field(default='--storePositions --storeDocvectors --storeRaw', metadata={"help":'pyserini index options'})

    threads: int = field(default=1, metadata={"help":'num threads'})


@dataclass
class SearchArguments():
    index_path: str = field(default=None, metadata={"help":"Path to the index directory location"})

    queries_path: str = field(default=None, metadata={"help":"Path to the tsv file where each line is in format 'id\tquery'"})

    nhits: int = field(default=10, metadata={"help":"Number of hits to return"})

    use_bm25: bool = field(default=True, metadata={"help":"Use bm25 scoring"})
    
    k1: float = field(default=0.8, metadata={"help":"bm25 parameter to tune impact of term frequency"})
    
    b: float = field(default=0.4, metadata={"help":"bm25 constant to fine tune the effect of document length"})

    output_dir: str = field(default=None, metadata={"help":"Output directory to write out search results"})

@dataclass
class BM25Config(SearchArguments, IndexingArguments):
    pass
