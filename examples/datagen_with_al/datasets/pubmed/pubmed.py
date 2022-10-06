# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PubMed abstracts dataset."""

import gzip
import xml.etree.ElementTree as ET

import datasets

_DESCRIPTION = """\
Abstracts from the PubMed database.
"""

_HOMEPAGE = "https://pubmed.ncbi.nlm.nih.gov/"


class PubmedConfig(datasets.BuilderConfig):
    """BuilderConfig for PubMed."""

    def __init__(self, num_files, **kwargs):
        """BuilderConfig for SuperGLUE.

        Args:
        num_files_to_download: int, the number of files to be downloaded,
        **kwargs: keyword arguments forwarded to super.
        """
        super(PubmedConfig, self).__init__(**kwargs)
        self.num_files = int(num_files)


class Pubmed(datasets.GeneratorBasedBuilder):
    """PubMed abstracts dataset."""

    VERSION = datasets.utils.Version("1.0.0")
    BUILDER_CONFIG_CLASS = PubmedConfig
    # BUILDER_CONFIGS = [
    #     PubmedConfig(
    #         version=VERSION,
    #         num_files=20,
    #     )
    # ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            # SQuAD format
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""

        # depending von the config download the files
        urls = [f"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed22n{str(i).rjust(4, '0')}.xml.gz" for i in range(1, self.config.num_files + 1)]
        cached_urls = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split("abstracts"),
                gen_kwargs={
                    "files": cached_urls,
                },
            ),
        ]

    def _generate_examples(self, files):
        """Yields examples."""

        for filename in files:
            with gzip.open(filename, 'rb') as f:
                # we downloaded only valid xml files, so all of them should be parseable
                tree = ET.parse(f)
            root = tree.getroot()

            for elem in root.iter('PubmedArticle'):
                # abstract is text attribute of subelement AbstractText
                abstract = elem.findtext('.//AbstractText')
                if abstract is None:
                    continue
                id = elem.findtext('.//PMID')
                yield id, {
                        "id": id,
                        "context": abstract,
                    }
