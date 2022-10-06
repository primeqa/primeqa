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
"""This script provides functionality for loading the TechQA dataset in two different configurations: with document retrieval and unanswerable questions (full) and answerable questions only with skipped document retrieval (RC)."""

import bz2
import json
import os
import random
from typing import Union

import datasets

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@misc{castelli2019techqa,
      title={The TechQA Dataset}, 
      author={Vittorio Castelli and Rishav Chakravarti and Saswati Dana and Anthony Ferritto and Radu Florian and Martin Franz and Dinesh Garg and Dinesh Khandelwal and Scott McCarley and Mike McCawley and Mohamed Nasr and Lin Pan and Cezar Pendus and John Pitrelli and Saurabh Pujar and Salim Roukos and Andrzej Sakrajda and Avirup Sil and Rosario Uceda-Sosa and Todd Ward and Rong Zhang},
      year={2019},
      eprint={1911.02984},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
We introduce TechQA, a domain-adaptation question answering dataset for the technical support domain. The TechQA corpus highlights two real-world issues from the automated customer support domain. First, it contains actual questions posed by users on a technical forum, rather than questions generated specifically for a competition or a task. Second, it has a real-world size -- 600 training, 310 dev, and 490 evaluation question/answer pairs -- thus reflecting the cost of creating large labeled datasets with actual data. Consequently, TechQA is meant to stimulate research in domain adaptation rather than being a resource to build QA systems from scratch. The dataset was obtained by crawling the IBM Developer and IBM DeveloperWorks forums for questions with accepted answers that appear in a published IBM Technote---a technical document that addresses a specific technical issue. We also release a collection of the 801,998 publicly available Technotes as of April 4, 2019 as a companion resource that might be used for pretraining, to learn representations of the IT domain language.
"""

_HOMEPAGE = "https://leaderboard.techqa.us-east.containers.appdomain.cloud/"

_LICENSE = "Community Data License Agreement – Permissive – Version 1.0"


class TechQAConfig(datasets.BuilderConfig):

    def __init__(self, answer_only: Union[bool, str] = True, **kwargs):
        """BuilderConfig for TechQA.

        Args:
        num_files_to_download: int, the number of files to be downloaded,
        **kwargs: keyword arguments forwarded to super.
        """
        super(TechQAConfig, self).__init__(**kwargs)
        self.answer_only = answer_only

    @property
    def answer_only(self):
        return self._answer_only

    @answer_only.setter
    def answer_only(self, value):
        if isinstance(value, str):
            if value in ['True', 'true', '1']:
                self._answer_only = True
            elif value in ['False', 'false', '0']:
                self._answer_only = False
            else:
                raise ValueError("Config argument `answer_only` has to be of type boolean or one of `True`, `true`, `False`, `False`")
        else:
            self._answer_only = bool(value)

class TechQA(datasets.GeneratorBasedBuilder):
    # TODO implement full config (incuding unanswerable samples and all contexts per sample)
    """A domain-adaptation question answering dataset for the technical support domain."""

    VERSION = datasets.Version("1.0.0")

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="full", version=VERSION, description="This config covers the whole dataset including document retrieval."),
        TechQAConfig(name="rc", version=VERSION, description="This config covers only the reading comprehension (RC) part in a SQuAD style."),
        datasets.BuilderConfig(name="technotes", version=VERSION, description="This config covers loading all documents (800K+)."),
    ]

    # This dataset requires users to download the data manually
    @property
    def manual_download_instructions(self) -> str:
        return f"Download the dataset at {_HOMEPAGE} and point to it by using the options `data_files` (downloaded archive) or `data_dir` (extracted archive)"

    def _info(self):
        if self.config.name == 'technotes':
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                }
            )
        elif self.config.name == "full":
            raise NotImplementedError()
        else:
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        if self.config.data_dir is not None:
            # extracted folder specified
            data_dir = self.config.data_dir
        else:
            # archive file specified
            data_dir = dl_manager.download_and_extract(self.config.data_files)
            data_dir = os.path.join(data_dir, 'TechQA')
        
        # make sure to select data for the RC task
        new = 'training_and_dev' not in os.listdir(data_dir)

        if self.config.name == 'technotes':
            return [
                datasets.SplitGenerator(
                    name=datasets.Split('technotes'),
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": None,
                        "technotes_path": os.path.join(data_dir, f"technote{'s' if new else '_corpus'}", 'full_technote_collection.txt.bz2' if not new else 'documents.jsonl.bz2'),
                        "new_format": new
                    },
                )
            ]
        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, f"training_and_dev{'_MRC' if new else ''}", f"training{'_MRC' if new else '_Q_A'}.json"),
                        "technotes_path": os.path.join(data_dir, f"training_and_dev{'_MRC' if new else ''}", 'training_dev_technotes.json'),
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, f"training_and_dev{'_MRC' if new else ''}", f"dev{'_MRC' if new else '_Q_A'}.json"),
                        "technotes_path": os.path.join(data_dir, f"training_and_dev{'_MRC' if new else ''}", 'training_dev_technotes.json'),
                        "split": "dev",
                    },
                ),
            ]

    def _generate_examples(
        self, filepath: str, technotes_path: str, split: str = None, new_format: bool = None
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.
        if self.config.name == 'technotes':
            open_fn = bz2.BZ2File if technotes_path.endswith('.bz2') else open
            with open_fn(technotes_path, 'r') as f:
                if not new_format:
                    for line in f:
                        arr = json.loads(line)
                        for entry in arr:
                            if isinstance(entry, str):
                                continue
                            id_ = entry['id']
                            title = entry['title']
                            context = entry['text']
                            yield id_, {
                                "id": id_,
                                "title": title,
                                "context": context,
                            }
                else:
                    for line in f:
                        entry = json.loads(line)
                        id_ = entry['id']
                        title = entry['title'].strip()
                        context = entry['text']
                        yield id_, {
                            "id": id_,
                            "title": title,
                            "context": context
                        }
        else:
            with open(technotes_path, 'r') as f:
                documents = json.load(f)

            with open(filepath) as f:
                data = json.load(f)
                
            if self.config.name == "full":
                raise NotImplementedError()
            else:
                for obj in data:
                    if self.config.answer_only and obj['ANSWERABLE'] != 'Y':
                        # skip sample
                        continue
                    # TODO use question or document title?
                    title = obj['QUESTION_TITLE'].strip()
                    question = obj['QUESTION_TEXT'].strip()
                    id_ = obj['QUESTION_ID']
                    if obj['ANSWERABLE'] == 'Y':
                        doc = documents[obj['DOCUMENT']]
                        context = doc['text']
                        answer_starts = [int(obj['START_OFFSET'])]
                        answers = [context[int(obj['START_OFFSET']):int(obj['END_OFFSET'])]]
                    else:
                        # correct document not given
                        # TODO use all documents
                        doc = documents[random.choice(obj['DOC_IDS'])]
                        context = doc['text']
                        # no answer
                        answer_starts = []
                        answers = []
                    
                    yield id_, {
                        "id": id_,
                        "title": title,
                        "context": context,
                        "question": question,
                        "answers": {
                            "answer_start": answer_starts,
                            "text": answers,
                        },
                    }

if __name__ == "__main__":
    CACHE_DIR = '~/arbeitsdaten/cache'
    data = datasets.load_dataset(__file__, 'technotes', data_dir=os.path.abspath(os.path.expanduser('~/arbeitsdaten/data/TechQA')), cache_dir=CACHE_DIR)
    # data = datasets.load_dataset(__file__, 'technotes', data_files=os.path.abspath(os.path.expanduser('~/Downloads/TechQA.tar.bz2')), cache_dir=CACHE_DIR)
    # data = datasets.load_dataset(__file__, 'technotes', cache_dir=CACHE_DIR)
    # data = datasets.load_dataset(__file__, 'rc', data_dir=os.path.abspath(os.path.expanduser('~/arbeitsdaten/data/TechQA')), cache_dir=CACHE_DIR)
    # data = datasets.load_dataset(__file__, 'rc', data_files=os.path.abspath(os.path.expanduser('~/Downloads/TechQA.tar.bz2')), cache_dir=CACHE_DIR)
    print(data)
    print(data['technotes'][0])
