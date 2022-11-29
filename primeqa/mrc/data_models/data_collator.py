from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch


@dataclass
class FiDDataCollator:
    r"""
    Modified from DataCollatorForSeq2Seq
    Do not pad the features for FID 
    The padding has been done in the preprocessor
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)


        batch_features = {}
        for feature in features:
            for k,v in feature.items(): 
                if k not in batch_features:
                    batch_features[k] = []
                batch_features[k].append(v)
        for k,v in batch_features.items():
            try: # not converting string features such as "query and "example_id" FIXME test this
                batch_features[k] = torch.tensor(v) # convert to tensor
            except:
                continue

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch_features["labels"])
            # features["decoder_input_ids"] = decoder_input_ids
            batch_features["decoder_input_ids"] = decoder_input_ids
        return batch_features