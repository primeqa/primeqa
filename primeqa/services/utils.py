import os
import json
import uuid


def generate_id() -> str:
    """
    Generate a unique identifier based on uuid4.

    Returns
    -------
    str:
        a unique identifier based on uuid4.

    """
    return str(uuid.uuid4())


def load_json(file_path: str, encoding: str = "utf-8"):
    """
    Load JSON file from the filesystem.

    Parameters
    ----------
    file_path: str
        path from where JSON file has to be loaded.
    encoding: str
        encoding used to load file (default: "utf-8")

    Returns
    -------

    """
    with open(file_path, "r", encoding=encoding) as file:
        item = json.load(file)
        return item


def save_json(item, file_path: str, encoding: str = "utf-8"):
    """
    Save an object to a JSON file.

    Parameters
    ----------
    item: object
        data to be stored into a JSON file.

    file_path: str
        path to JSON file

    encoding: str
        encoding used to save file (default: "utf-8")

    Returns
    -------

    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding=encoding) as file:
        json.dump(item, file, indent=4)
