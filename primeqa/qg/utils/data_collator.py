from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import DataCollatorForSeq2Seq


@dataclass
class T2TDataCollator:
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example["input_ids"] for example in batch])
        lm_labels = torch.stack([example["target_ids"] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        decoder_attention_mask = torch.stack(
            [example["target_attention_mask"] for example in batch]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_attention_mask": decoder_attention_mask,
        }


@dataclass
class DataCollatorForSeq2SeqWithDecoderInputs(DataCollatorForSeq2Seq):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels and decoder_input_ids.
    """

    def __call__(self, features, **kwargs):
        decoder_input_ids = (
            [feature["decoder_input_ids"] for feature in features]
            if "decoder_input_ids" in features[0].keys()
            else None
        )
        # We have to pad the decoder_input_ids before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if decoder_input_ids is not None:
            max_label_length = max(len(l) for l in decoder_input_ids)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                    max_label_length - len(feature["decoder_input_ids"])
                )
                feature["decoder_input_ids"] = (
                    feature["decoder_input_ids"] + remainder
                    if padding_side == "right"
                    else remainder + feature["decoder_input_ids"]
                )

        return super().__call__(features, **kwargs)
