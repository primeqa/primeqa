from typing import List
import os
import shutil
from pathlib import Path


from primeqa.services.utils import generate_id, load_json, save_json


DIR_NAME_INDEXES = "indexes"
FILENAME_INFORMATION = "information"
FILENAME_DOCUMENTS = "documents"
FILENAME_DOCUMENT_IDS = "document_ids"
FILENAME_MODEL = "model.dnn"
EXTN_JSON = ".json"
EXTN_TSV = ".tsv"
EXTN_TXT = ".txt"


#############################################################################################
# indexes/
#        <index-id>/
#                   details.json
#                   documents.json
#                   documents.tsv
# models/
#        <model-id>/
#                   details.json
#                   data.json

#############################################################################################
class Store:
    def __init__(self):
        self.root_dir = os.getenv(
            "STORE_DIR", os.path.join(Path(__file__).parent.parent.parent, "store")
        )
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    #############################################################################################
    #                       Index Documents
    #############################################################################################
    def get_index_document_texts_file_path(self, index_id: str):
        return os.path.join(
            self.root_dir,
            DIR_NAME_INDEXES,
            str(index_id),
            FILENAME_DOCUMENTS + EXTN_TSV,
        )

    def get_index_document_ids_file_path(self, index_id: str):
        return os.path.join(
            self.root_dir,
            DIR_NAME_INDEXES,
            str(index_id),
            FILENAME_DOCUMENT_IDS + EXTN_TXT,
        )

    def get_index_document_texts(self, index_id: str):
        documents = list()
        with open(
            self.get_index_document_texts_file_path(index_id), "r", encoding="utf-8"
        ) as file_pointer:
            for line in file_pointer.readlines():
                documents.append(line.rstrip("\n").split("\t")[-1])

        return documents

    def get_index_document_ids(self, index_id: str):
        document_ids = list()
        with open(
            self.get_index_document_ids_file_path(index_id), "r", encoding="utf-8"
        ) as file_pointer:
            for line in file_pointer.readlines():
                document_ids.append(line.rstrip("\n"))

        return document_ids

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
        Retrieves index id for all indexes.

        Returns
        -------
        list:
            index ids

        """
        return [
            index_dir.stem
            for index_dir in Path(os.path.join(self.root_dir, DIR_NAME_INDEXES)).glob(
                "*"
            )
            if os.path.isdir(index_dir)
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

    def save_documents(self, index_id: str, documents: List[dict]):
        """
        Save documents used for indexing

        Parameters
        ----------
        index_id: str
            unique identifier for the index
        documents: List
            document used for indexing

        Returns
        -------
        """
        documents_file_path = self.get_index_document_texts_file_path(index_id)
        os.makedirs(os.path.dirname(documents_file_path), exist_ok=True)

        document_ids_file_path = self.get_index_document_ids_file_path(index_id)
        os.makedirs(os.path.dirname(document_ids_file_path), exist_ok=True)

        with open(documents_file_path, "w", encoding="utf-8") as documents_file, open(
            document_ids_file_path, "w", encoding="utf-8"
        ) as document_ids_file:
            for document_idx, document in enumerate(documents):
                documents_file.write(str(document_idx) + "\t" + document["text"] + "\n")
                document_ids_file.write(document["document_id"] + "\n")


class StoreFactory:
    _instance = None

    @classmethod
    def get_store(cls) -> Store:
        if not cls._instance:
            cls._instance = Store()

        return cls._instance
