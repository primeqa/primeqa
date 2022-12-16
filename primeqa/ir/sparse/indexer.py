import logging
import os
import shutil
import tempfile
import subprocess
import json

from tqdm import tqdm
from pyserini.search import LuceneSearcher

from primeqa.ir.util.corpus_reader import corpus_reader


logger = logging.getLogger(__name__)


class PyseriniIndexer:
    """
    A class to handle indexing a collection of documents in Pyserini
    """

    def __init__(self):
        pass

    def _clean_text(self, text: str):
        return text.replace("\t", " ")

    def _run_command(self, cmd):
        logger.info(cmd)
        process = subprocess.Popen(cmd.split())
        rc = process.wait()
        return rc

    def _preprocess_corpus(self, collection, tmpdirname, fieldnames=None):
        reader = corpus_reader(collection, fieldnames=fieldnames)
        num_docs = 0
        with open(
            os.path.join(tmpdirname, "corpus_pyserini_fmt.jsonl"),
            "w",
            encoding="utf-8",
        ) as outf:
            for passage in tqdm(reader):
                json_string = json.dumps(
                    {
                        "id": passage.pid,
                        "contents": f"{self._clean_text(passage.title)}\t{self._clean_text(passage.text)}",
                    }
                )
                outf.write(f"{json_string}\n")
                num_docs += 1
        return num_docs

    def index_collection(
        self,
        collection: str,
        index_path: str,
        fieldnames=None,
        overwrite=False,
        threads=1,
        additional_index_cmd_args="--storePositions --storeDocvectors --storeRaw",
    ):
        """

        Index the corpus of documents.
        - First convert the input corpus to the json format requiered by Pyserini 'DefaultLuceneDocumentGenerator'.
        - This will write to a temporary directory within the directory specified by the 'index_path' argument
        - Second run the indexing command.  This launches a subprocess and runs  'python -m pyserini.index.lucene <args>'
        - Validate the index is usable by opening the index and checking the the number of documents
        is equal to the intput corpus.

        Args:
            collection (str) : path to file or directory of documents in tsv or jsonl format.
            index_path (str) : output directory path where the index is written
            fieldnames ( List, Optional): column headers to be assigned to tsv without headers
            overwrite (bool, Optional): overwrite an existing directory, defaults to false
            threads (int): num threads to be used when indexing
            additional_index_cmd_args (str, Optional): indexing arguments, defaults to '--storePositions --storeDocvectors --storeRaw'

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if (
            not overwrite
            and os.path.exists(index_path)
            and os.listdir(index_path)
        ):
            raise ValueError(
                f"Index path not empty '{index_path}' and overwrite not specified"
            )
        if not os.path.exists(index_path):
            os.makedirs(index_path)
        # create temporary subdirectory for the corpus
        with tempfile.TemporaryDirectory(
            prefix="tmp", dir=index_path
        ) as tmpdirname:
            # convert corpus documents to pyserini jsonl
            num_docs = self._preprocess_corpus(
                collection, tmpdirname, fieldnames=fieldnames
            )
            # build index command
            cmd1 = (
                "python -m pyserini.index.lucene -collection JsonCollection "
                + "-generator DefaultLuceneDocumentGenerator "
                + f"-threads {threads}  {additional_index_cmd_args} "
                f"-input {tmpdirname} -index {index_path}"
            )
            # run the command
            rc = self._run_command(cmd1)
            # cleanup temporary corpus directory
            shutil.rmtree(tmpdirname)
        assert rc == 0

        logger.info(
            "Indexing completed at index location %s. validating document count",
            index_path,
        )
        searcher = LuceneSearcher(index_path)
        logger.info(
            "Index %s contains %d documents", index_path, searcher.num_docs
        )
        assert searcher.num_docs == num_docs
        logging.info("Index available at %s", index_path)
        searcher.close()
        return rc
