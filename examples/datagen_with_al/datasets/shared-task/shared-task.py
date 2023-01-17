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
"""MRQA 2019 Shared task dataset."""

import json

import datasets

_CITATION = """\
@inproceedings{fisch2019mrqa,
    title={{MRQA} 2019 Shared Task: Evaluating Generalization in Reading Comprehension},
    author={Adam Fisch and Alon Talmor and Robin Jia and Minjoon Seo and Eunsol Choi and Danqi Chen},
    booktitle={Proceedings of 2nd Machine Reading for Reading Comprehension (MRQA) Workshop at EMNLP},
    year={2019},
}
"""

_DESCRIPTION = """\
The MRQA 2019 Shared Task focuses on generalization in question answering.
An effective question answering system should do more than merely
interpolate from the training set to answer test examples drawn
from the same distribution: it should also be able to extrapolate
to out-of-distribution examples — a significantly harder challenge.

The dataset is a collection of 18 existing QA dataset (carefully selected
subset of them) and converted to the same format (SQuAD format). Among
these 18 datasets, six datasets were made available for training,
six datasets were made available for development, and the final six
for testing. The dataset is released as part of the MRQA 2019 Shared Task.
"""

_HOMEPAGE = "https://mrqa.github.io/2019/shared.html"

_LICENSE = "Unknwon"

_URLs = {
    # Train+Dev sub-datasets
    "squad": {
        "train+SQuAD": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz",
        "validation+SQuAD": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz",
    },
    "newsqa": {
        "train+NewsQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz",
        "validation+NewsQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz",
    },
    "triviaqa": {
        "train+TriviaQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/TriviaQA-web.jsonl.gz",
        "validation+TriviaQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz",
    },
    "searchqa": {
        "train+SearchQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SearchQA.jsonl.gz",
        "validation+SearchQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz",
    },
    "hotpotqa": {
        "train+HotpotQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/HotpotQA.jsonl.gz",
        "validation+HotpotQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz",
    },
    "naturalquestions": {
        "train+NaturalQuestions": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz",
        "validation+NaturalQuestions": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz",
    },
    # Test sub-datasets
    "bioasq": {
        "test+BioASQ": "http://participants-area.bioasq.org/MRQA2019/",  # BioASQ.jsonl.gz
    },
    "drop": {
        "test+DROP": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DROP.jsonl.gz",
    },
    "duorc": {
        "test+DuoRC": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DuoRC.ParaphraseRC.jsonl.gz",
    },
    "race": {
        "test+RACE": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RACE.jsonl.gz",
    },
    "relationextraction": {
        "test+RelationExtraction": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RelationExtraction.jsonl.gz",
    },
    "textbookqa": {
        "test+TextbookQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TextbookQA.jsonl.gz",
    },
}


class Mrqa(datasets.GeneratorBasedBuilder):
    """MRQA 2019 Shared task dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="squad", description="SQuAD", version=VERSION),
        datasets.BuilderConfig(name="newsqa", description="NewsQa", version=VERSION),
        datasets.BuilderConfig(
            name="triviaqa", description="TriviaQA", version=VERSION
        ),
        datasets.BuilderConfig(
            name="searchqa", description="SearchQA", version=VERSION
        ),
        datasets.BuilderConfig(
            name="hotpotqa", description="HotpotQA", version=VERSION
        ),
        datasets.BuilderConfig(
            name="naturalquestions", description="NaturalQuestions", version=VERSION
        ),
        datasets.BuilderConfig(name="bioasq", description="BioASQ", version=VERSION),
        datasets.BuilderConfig(name="drop", description="DROP", version=VERSION),
        datasets.BuilderConfig(name="duorc", description="DuoRC", version=VERSION),
        datasets.BuilderConfig(name="race", description="RACE", version=VERSION),
        datasets.BuilderConfig(
            name="relationextraction", description="RelationExtraction", version=VERSION
        ),
        datasets.BuilderConfig(
            name="textbookqa", description="TextbookQA", version=VERSION
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            # SQuAD format
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.Sequence(
                        {
                            "text": datasets.Sequence(datasets.Value("string")),
                            "answer_start": datasets.Sequence(datasets.Value("int32")),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URLs[self.config.name])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths_dict": data_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths_dict": data_dir,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepaths_dict": data_dir,
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepaths_dict, split):
        """Yields examples."""
        for source, filepath in filepaths_dict.items():
            if split not in source:
                continue
            with open(filepath, encoding="utf-8") as f:
                header = next(f)
                # subset = json.loads(header)["header"]["dataset"]

                for row in f:
                    paragraph = json.loads(row)
                    context = paragraph["context"]
                    for qa in paragraph["qas"]:
                        qid = qa["qid"]
                        question = qa["question"].strip()
                        answers_start = []
                        answers_text = []
                        # instead of mixing answers and their occurences keep them separate, i.e. a list of list of spans (one list of spans per answer)
                        for detect_ans in qa["detected_answers"]:
                            cur_answers_start = []
                            cur_answers_text = []
                            for char_span in detect_ans["char_spans"]:
                                # answers_text.append(detect_ans["text"].strip())
                                # detected answers text are wrong sometimes, rely on char span instead
                                cur_answers_text.append(
                                    context[char_span[0] : char_span[1] + 1].strip()
                                )
                                cur_answers_start.append(char_span[0])
                            answers_start.append(cur_answers_start)
                            answers_text.append(cur_answers_text)
                        yield f"{source}_{qid}", {
                            "id": qid,
                            "context": context,
                            "question": question,
                            "answers": {
                                "answer_start": answers_start,
                                "text": answers_text,
                            },
                        }
