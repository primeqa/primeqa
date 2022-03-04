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

_deps = {
    "bump2version": ["dev"],
    "datasets>=1.4.0": ["install"],
    "pytest": ["tests"],
    "pytest-cov": ["tests"],
    "pytest-rerunfailures": ["tests"],
    "torch>=1.8": ["install"],  # TODO: see if we can reduce to 1.7 or 1.6
    "tox": ["tests"],
    "transformers>=4.0.0": ["install"],
    "tqdm": ["install"],
}

extras_names = ["dev", "install", "tests"]
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
    extras_require=extras,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "TODO"
    ],
)
