from typing import List
import os
import shutil
from pathlib import Path
import glob

from cachetools.func import ttl_cache
from sqlitedict import SqliteDict

from primeqa.services.utils import generate_id, load_json, save_json


DIR_NAME_INDEXES = "indexes"
DIR_NAME_INDEX = "index"
DIR_NAME_MODELS = "models"
DIR_NAME_CHECKPOINTS = "checkpoints"
DIR_NAME_MODELS = "models"
FILENAME_INFORMATION = "information"
FILENAME_DOCUMENTS = "documents"
FILENAME_DOCUMENT_IDS = "document_ids"
FILENAME_MODEL = "model.dnn"
EXTN_JSON = ".json"
EXTN_TSV = ".tsv"
EXTN_TXT = ".txt"
EXTN_SQL_LITE = ".sqlite"

#############################################################################################
# indexes/
#        <index-id>/
#                   details.json
#                   documents.json
#                   documents.tsv
# checkpoints/
#            <checkpoint>/
#                       <model-id>.dnn|<model-id>.model
# models/
#        <model-id>/
#               *.dnn|*.model
#############################################################################################
class Store:
    def __init__(self):
        self.root_dir = os.getenv(
            "STORE_DIR", os.path.join(Path(__file__).parent.parent.parent, "store")
        )
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        # Check if indexes directory exist, if not create one
        if not os.path.exists(os.path.join(self.root_dir, DIR_NAME_INDEXES)):
            os.makedirs(os.path.join(self.root_dir, DIR_NAME_INDEXES))

        # Check if checkpoints directory exist, if not create one
        if not os.path.exists(os.path.join(self.root_dir, DIR_NAME_CHECKPOINTS)):
            os.makedirs(os.path.join(self.root_dir, DIR_NAME_CHECKPOINTS))

        # Check if models directory exist, if not create one
        if not os.path.exists(os.path.join(self.root_dir, DIR_NAME_MODELS)):
            os.makedirs(os.path.join(self.root_dir, DIR_NAME_MODELS))

    def exists(self, path: str):
        return os.path.exists(path)

    #############################################################################################
    #                       Checkpoints
    #############################################################################################
    def get_checkpoint_path(self, checkpoint: str):
        return glob.glob(
            f"{os.path.join(self.root_dir, DIR_NAME_CHECKPOINTS, str(checkpoint))}/*"
        )[0]

    #############################################################################################
    #                       models
    #############################################################################################
    def get_index_documents_file_path(self, index_id: str, extension: str = EXTN_TSV):
        return os.path.join(
            self.root_dir,
            DIR_NAME_INDEXES,
            str(index_id),
            f"{FILENAME_DOCUMENTS}{extension}",
        )

    @ttl_cache(maxsize=10, ttl=10 * 60)
    def get_index_documents_database(self, index_id: str):
        return SqliteDict(
            self.get_index_documents_file_path(index_id, extension=EXTN_SQL_LITE),
            tablename="documents",
        )

    def get_index_document(self, index_id: str, document_idx: int):
        database = self.get_index_documents_database(index_id=index_id)
        return database[str(document_idx)]

    def save_index_documents(self, index_id: str, documents: List[dict]):
        # Step 1: Create `documents.tsv` in index directory
        documents_tsv_file_path = self.get_index_documents_file_path(
            index_id, extension=EXTN_TSV
        )
        os.makedirs(os.path.dirname(documents_tsv_file_path), exist_ok=True)

        # Step 2: Create `documents.sqlite` in index directory
        documents_sqlite_file_path = self.get_index_documents_file_path(
            index_id, extension=EXTN_SQL_LITE
        )
        os.makedirs(os.path.dirname(documents_sqlite_file_path), exist_ok=True)

    def get_index_document(self, index_id: str, document_idx: int):
        database = self.get_index_documents_database(index_id=index_id)
        return database[str(document_idx)]

    def save_index_documents(self, index_id: str, documents: List[dict]):
        # Step 1: Create `documents.tsv` in index directory
        documents_tsv_file_path = self.get_index_documents_file_path(
            index_id, extension=EXTN_TSV
        )
        os.makedirs(os.path.dirname(documents_tsv_file_path), exist_ok=True)

        # Step 2: Create `documents.sqlite` in index directory
        documents_sqlite_file_path = self.get_index_documents_file_path(
            index_id, extension=EXTN_SQL_LITE
        )
        os.makedirs(os.path.dirname(documents_sqlite_file_path), exist_ok=True)

        # Step 3: Iterate over documents in the request to save to `documents.tsv` and `documents.sqlite`
        with open(
            documents_tsv_file_path, "w", encoding="utf-8"
        ) as documents_file, SqliteDict(
            documents_sqlite_file_path, tablename="documents"
        ) as documents_db:
            # Step 3.a: Add heading row to `documents.tsv`
            documents_file.write("id\ttext\ttitle\n")

            # Step 3.b: Iterate over documents to add rows to `documents.tsv` and entries in documents_db
            for document_idx, document in enumerate(documents):
                documents_file.write(
                    f"{str(document_idx + 1)}\t{document['text']}\t{document['title'] if 'title' in document else ''}\n"
                )
                documents_db[str(document_idx + 1)] = document

            # Step 3.c: Commit to save documents_db
            documents_db.commit()

        # Step 4: Clear cache
        self.get_index_documents_database.cache_clear()

    #############################################################################################
    #                       Indexes
    #############################################################################################
    def generate_index_uuid(self) -> str:
        already_in_use_ids = set(self.get_index_ids())
        # Generate UUID4 and check against existing ids
        uuid = generate_id()
        while uuid in already_in_use_ids:
            uuid = generate_id()

        return uuid

    def get_index_ids(self) -> List[str]:
        """
        Get list of all index ids.

        Returns:
            List[str]: list of index ids
        """

        return [
            index_dir.stem
            for index_dir in Path(os.path.join(self.root_dir, DIR_NAME_INDEXES)).glob(
                "*"
            )
            if os.path.isdir(index_dir)
        ]

    def get_indexes(self) -> List[dict]:
        return [
            self.get_index_information(index_id=index_id)
            for index_id in self.get_index_ids()
        ]

    def get_index_directory_path(self, index_id: str) -> str:
        """
        Get directory path of an index.

        Parameters
        ----------
        index_id: str
            unique identifier for the index.

        Returns
        -------
        str:
            index's directory.

        """
        return os.path.join(self.root_dir, DIR_NAME_INDEXES, str(index_id))

    def get_index_file_path(self, index_id: str) -> str:
        """
        Get path to index in indexs/<index_id> directory.

        Parameters
        ----------
        index_id: str
            unique identifier for the index.

        Returns
        -------
        str:
            index's file path.

        """
        return os.path.join(
            self.root_dir, DIR_NAME_INDEXES, str(index_id), DIR_NAME_INDEX
        )

    def delete_index(self, index_id: str) -> None:
        """
        Delete specified index.

        Parameters
        ----------
        index_id: str
            unique identifier for the index.

        Returns
        -------
        """
        index_dir_to_be_deleted = self.get_index_directory_path(index_id)
        if os.path.exists(index_dir_to_be_deleted):
            shutil.rmtree(index_dir_to_be_deleted)

    def get_index_information(self, index_id: str) -> dict:
        """
        Get index information.

        Parameters
        ----------
        index_id: str
            unique identifier for the index.

        Returns
        -------
        dict:
            index's information.

        """
        return load_json(
            os.path.join(
                self.root_dir,
                DIR_NAME_INDEXES,
                str(index_id),
                FILENAME_INFORMATION + EXTN_JSON,
            )
        )

    def save_index_information(self, index_id: str, information: dict) -> None:
        """
        Save index information.

        Parameters
        ----------
        index_id: str
            unique identifier for the index
        information: dict
            index information

        Returns
        -------
        """
        return save_json(
            information,
            file_path=os.path.join(
                self.root_dir,
                DIR_NAME_INDEXES,
                str(index_id),
                FILENAME_INFORMATION + EXTN_JSON,
            ),
        )


class StoreFactory:
    _instance = None

    @classmethod
    def get_store(cls) -> Store:
        if not cls._instance:
            cls._instance = Store()

        return cls._instance
