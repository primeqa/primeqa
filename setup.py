import os
import re
from itertools import chain

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    long_description = readme_file.read()

# Get the version for the project
with open(os.path.join(here, 'VERSION')) as version_file:
    version = version_file.read().strip()

include_package_roots = ["oneqa"]  # only include these packages and their subpackages
include_packages = list(chain.from_iterable(map(lambda name: [name, f"{name}.*"], include_package_roots)))

keywords = [
    "NLP", "transformers", "QA", "question", "answering", "mrc", "rc", "machine", "reading", "comprehension",
    "IR", "information", "retrieval", "deep", "learning", "pytorch", "BERT", "RoBERTa"
]

authors = [
    "TODO"
]

_deps = [
    "bump2version",
    "pytest",
    "torch>=1.8",  # TODO: see if we can reduce to 1.7 or 1.6
    "tox",
    "tox-travis",
    "transformers"  # TODO change this to range and add sentencepiece
]

# this is a lookup table with items like:
#
# tokenizers: "tokenizers==0.9.4"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
# deps = {b: a for a, b in (re.findall(r"^(([^!=<>]+)(?:[!=<>].*)?$)", x)[0] for x in _deps)}


# def deps_list(*pkgs):
#     return [deps[pkg] for pkg in pkgs]


extras = {}

# TODO allow minimal set of dependencies to be installed


install_requires = _deps

setup(
    name="one-qa",
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
    # dev_requires=None,
    # test_requires=None,
    # docs_require=None,
    # extras_require=extras,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "TODO"
    ],
)
