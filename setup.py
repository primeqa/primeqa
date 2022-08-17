import os
import re
import sys
from itertools import chain

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    long_description = readme_file.read()

# Get the version for the project
with open(os.path.join(here, 'VERSION')) as version_file:
    version = version_file.read().strip()

include_package_roots = ["primeqa"]  # only include these packages and their subpackages
include_packages = list(chain.from_iterable(map(lambda name: [name, f"{name}.*"], include_package_roots)))

keywords = [
    "NLP", "transformers", "QA", "question", "answering", "mrc", "rc", "machine", "reading", "comprehension",
    "IR", "information", "retrieval", "deep", "learning", "pytorch", "BERT", "RoBERTa", "T5", "generation", "table"
]

authors = [
    "TODO"
]

_deps = {
    "docutils>=0.14,<0.18": ["tests"],
    "bitarray~=2.3.7": ["install"],
    "bump2version~=1.0.1": ["dev"],
    "click~=8.0.4": ["install"],
    "datasets~=2.4.0": ["install"],
    "myst-parser~=0.17.2": ["docs"],
    "faiss-cpu~=1.7.2": ["install"],
    "faiss-gpu~=1.7.2": ["install"],
    "gitpython~=3.1.27": ["install"],
    "ipykernel~=6.13.0": ["notebooks"],
    "ipywidgets~=7.7.0": ["notebooks"],
    "jsonlines~=3.0.0": ["install"],
    "ninja~=1.10.2.3": ["install"],
    "nltk~=3.7": ["install"],
    "numpy~=1.21.5": ["install"],
    "myst-parser~=0.17.2": ["docs"],
    "packaging~=21.3": ["install"],
    "pandas~=1.3.5": ["install"],
    "psutil~=5.9.0": ["install"],
    "pydata-sphinx-theme~=0.8.0": ["docs"],
    "pyserini~=0.16.0": ["install"],
    "pytest~=7.1.1": ["tests"],
    "pytest-cov~=3.0.0": ["tests"],
    "pytest-mock~=3.7.0": ["tests"],
    "pytest-rerunfailures~=10.2": ["tests"],
    "scikit-learn~=1.0.2": ["install"],
    "signals~=0.0.2": ["install"],
    "spacy~=3.2.2": ["install"],
    "stanza~=1.4.0":["install"],
    "sphinx~=4.4.0": ["docs"],
    "torch~=1.11.0": ["install"],
    "tox~=3.24.5": ["tests"],
    "transformers~=4.17.0": ["install"],
    "sentencepiece~=0.1.96": ["install"],
    "protobuf~=3.20.0": ["install"],
    "tqdm~=4.64.0": ["install"],
    "transformers~=4.17.0": ["install"],
    "ujson~=5.1.0": ["install"],
    "transformers~=4.17.0": ["install"],
    "tqdm~=4.64.0": ["install"],
    # "torch-scatter @ https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl": ["install"],
    "frozendict": ["install"],
    "nlp": ["install"],
    "sentencepiece~=0.1.96": ["install"],
    "protobuf~=3.20.0": ["install"],
    "nltk~=3.6":["install"],
    "tabulate~=0.8.9":["install"],
    "rouge_score":["install"]

}

extras_names = ["docs", "dev", "install", "notebooks", "tests"]
extras = {extra_name: [] for extra_name in extras_names}
for package_name, package_required_by in _deps.items():
    if not package_required_by:
        raise ValueError(f"Package '{package_name}' has no associated extras it is required for. "
                         f"Choose one or more of: {extras_names}")

    for extra_name in package_required_by:
        try:
            extras[extra_name].append(package_name)
        except KeyError as ex:
            raise KeyError(f"Extras name '{extra_name}' not in {extras_names}") from ex
extras["all"] = list(_deps)

install_requires = extras["install"]

python_version = sys.version_info.major,sys.version_info.minor

"""
if python_version == (3,7):
    install_requires.append("torch-scatter @ https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl")
elif python_version == (3,8):
    install_requires.append("torch-scatter @ https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl")
elif python_version == (3,9):
    install_requires.append("torch-scatter @ https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl")
"""

setup(
    name="prime-qa",
    version=version,
    author=", ".join(authors),
    author_email="TODO",
    description="State-of-the-art Question Answering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="TODO",
    license="TODO",
    keywords=" ".join(keywords),
    packages=find_packages(".", include=include_packages),
    python_requires=">=3.7.0",
    install_requires=install_requires,
    extras_require=extras,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "TODO"
    ],
)
