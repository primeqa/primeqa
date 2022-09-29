import sys
import os
import platform

from itertools import chain
from setuptools import setup, find_packages

if sys.platform == "win32" and sys.maxsize.bit_length() == 31:
    print(
        "32-bit Windows Python runtime is not supported. Please switch to 64-bit Python."
    )
    sys.exit(-1)


python_min_version = (3, 7, 0)
python_max_version = (3, 10, 0)

if python_min_version > sys.version_info or python_max_version < sys.version_info:
    print(
        f"You are using Python {platform.python_version()}. {'.'.join(map(str, python_min_version))} >= Python <= {'.'.join(map(str, python_max_version))} is required."
    )
    sys.exit(-1)

################################################################################
# constants
################################################################################
cwd = os.path.dirname(os.path.abspath(__file__))


################################################################################
# Version, description and package_name
################################################################################
package_name = os.getenv("PRIMEQA_PACKAGE_NAME", "primeqa")
package_type = os.getenv("PACKAGE_TYPE", "wheel")
with open(os.path.join(cwd, "VERSION"), "r", encoding="utf-8") as version_file:
    version = version_file.read().strip()
with open(os.path.join(cwd, "README.md"), encoding="utf-8") as readme_file:
    long_description = readme_file.read()


################################################################################
# Build
################################################################################
print(f"Building wheel {package_name}-{version}")

include_packages = list(
    chain.from_iterable(map(lambda name: [name, f"{name}.*"], ["primeqa"]))
)

_deps = {
    "docutils>=0.14,<0.18": ["tests"],
    "bitarray~=2.3.7": ["install", "gpu"],
    "bump2version~=1.0.1": ["dev"],
    "click~=8.0.4": ["install", "gpu"],
    "datasets[apache-beam]~=2.3.2": ["install", "gpu"],
    "faiss-cpu~=1.7.2": ["install", "gpu"],
    "faiss-gpu~=1.7.2": ["gpu"],
    "gitpython~=3.1.27": ["install", "gpu"],
    "ipykernel~=6.13.0": ["notebooks"],
    "ipywidgets~=7.7.0": ["notebooks"],
    "jsonlines~=3.0.0": ["install", "gpu"],
    "ninja~=1.10.2.3": ["install", "gpu"],
    "nltk~=3.7": ["install", "gpu"],
    "numpy~=1.21.5": ["install", "gpu"],
    "packaging~=21.3": ["install", "gpu"],
    "pandas~=1.3.5": ["install", "gpu"],
    "psutil~=5.9.0": ["install", "gpu"],
    "pyserini~=0.16.0": ["install", "gpu"],
    "pytest~=7.1.1": ["tests"],
    "pytest-cov~=3.0.0": ["tests"],
    "pytest-mock~=3.7.0": ["tests"],
    "pytest-rerunfailures~=10.2": ["tests"],
    "scikit-learn~=1.0.2": ["install", "gpu"],
    "signals~=0.0.2": ["install", "gpu"],
    "spacy~=3.2.2": ["install", "gpu"],
    "stanza~=1.4.0": ["install", "gpu"],
    "torch~=1.11.0": ["install", "gpu"],
    "tox~=3.24.5": ["tests"],
    "transformers~=4.17.0": ["install", "gpu"],
    "sentencepiece~=0.1.96": ["install", "gpu"],
    "ujson~=5.1.0": ["install"],
    "tqdm~=4.64.0": ["install", "gpu"],
    "frozendict": ["install", "gpu"],
    "nlp": ["install", "gpu"],
    "protobuf~=3.20.0": ["install", "gpu"],
    "tabulate~=0.8.9": ["install", "gpu"],
    "rouge_score": ["install", "gpu"],
    "myst-parser~=0.17.2": ["docs"],
    "pydata-sphinx-theme~=0.9.0": ["docs"],
    "sphinx~=4.4.0": ["docs"],
    "sphinx_design~=0.2.0": ["docs"],
    "recommonmark~=0.7.1": ["docs"],
}

extras_names = ["docs", "dev", "install", "notebooks", "tests", "gpu"]
extras = {extra_name: [] for extra_name in extras_names}
for dep_package_name, dep_package_required_by in _deps.items():
    if not dep_package_required_by:
        raise ValueError(
            f"Package '{dep_package_name}' has no associated extras it is required for. "
            f"Choose one or more of: {extras_names}"
        )

    for extra_name in dep_package_required_by:
        try:
            extras[extra_name].append(dep_package_name)
        except KeyError as ex:
            raise KeyError(f"Extras name '{extra_name}' not in {extras_names}") from ex
extras["all"] = list(_deps)

install_requires = extras["install"]

setup(
    name=package_name,
    url="https://github.com/primeqa/primeqa",
    version=version,
    author="PrimeQA Team",
    author_email="primeqa@us.ibm.com",
    description="State-of-the-art Question Answering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache",
    keywords="Question Answering (QA), Machine Reading Comprehension (MRC), Information Retrieval (IR)",
    packages=find_packages(".", include=include_packages),
    python_requires=">=3.7.0, <3.10.0",
    install_requires=install_requires,
    extras_require=extras,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
)
