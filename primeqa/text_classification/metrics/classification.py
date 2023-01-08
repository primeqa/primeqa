# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors.
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
""" GLUE benchmark metric. """

from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

import datasets

_CITATION = """"""

_DESCRIPTION = """\
Boolean metrics
"""

_KWARGS_DESCRIPTION = """
Compute Boolean evaluation metrics.
Args:
    predictions: list of predictions to score.
        Each translation should be tokenized into a list of tokens.
    references: list of lists of references for each translation.
        Each reference should be tokenized into a list of tokens.
Returns: depending on the GLUE subset, one or several of:
    "accuracy": Accuracy
    "f1": F1 score
    "conf": Confusion Matrix
Examples:

    >>> glue_metric = datasets.load_metric('/path/to/this/file', 'qtype')  # 'qtype' or any of []
    >>> references = [0, 1]
    >>> predictions = [0, 1]
    >>> results = glue_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'accuracy': 1.0}
"""

def prf(preds, refs, labels, lang):
    precision, recall, f1, support = precision_recall_fscore_support(refs, preds, labels=labels)
    conf_matrix = confusion_matrix(refs, preds, labels=labels)

    return {
        lang + '_avg_f1': sum(f1)/len(f1),
        lang + '_f1': str(f1),
        lang + '_precision': str(precision),
        lang + '_recall': str(recall),
        lang + '_support': str(support),
        lang + '_confusion matrix': str(conf_matrix).replace("\n", " ")
    }

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Classification(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {
                        "example_id": datasets.Value("string"), 
                        "prediction": datasets.Value("string"),
                        "confidence": datasets.Value("float32"),
                        "language": datasets.Value("string"),
                        "question": datasets.Value("string"),
                        "scores": datasets.Value("string") 
                    },
                    "references": {
                        "example_id": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "language": datasets.Value("string"),
                        "question": datasets.Value("string")
                    },
                }
            )
        )

    def _compute(self, predictions, references):
        preds = []
        refs = []
        preds_by_lang = {}
        refs_by_lang = {}
        
        for prediction, reference in zip(predictions,references):
                
            preds.append(prediction['prediction'])
            refs.append(reference['label'])
            
            if 'language' not in reference:
                language = 'english'
            else:
                language = reference['language']
            if language not in preds_by_lang:
                preds_by_lang[language] = []
                refs_by_lang[language] = []
            preds_by_lang[language].append(prediction['prediction'])
            refs_by_lang[language].append(reference['label'])

        scores = {}
        print("labels")
        print(list(set(refs)))
        scores.update(prf(preds, refs, list(set(refs)), 'all'))
        for language in preds_by_lang:
            scores.update(prf(preds_by_lang[language], refs_by_lang[language], list(set(refs)), language))
        return scores