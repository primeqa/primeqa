import torch

from transformers import RobertaTokenizerFast
from primeqa.ir.dense.colbert_top.colbert.modeling.hf_colbert_roberta import HF_ColBERT_Roberta
from primeqa.ir.dense.colbert_top.colbert.modeling.tokenization.utils import _split_into_batches, _sort_by_length
from primeqa.ir.dense.colbert_top.colbert.utils.utils import print_message

class DocTokenizerRoberta():
    def __init__(self, doc_maxlen, model_type):
        # self.tok = XLMRobertaTokenizer.from_pretrained(model_type)
        self.tok = HF_ColBERT_Roberta.raw_tokenizer_from_pretrained(model_type)

        self.doc_maxlen = doc_maxlen

        self.D_marker_token, self.D_marker_token_id = '[D]', self.tok.convert_tokens_to_ids('madeupword0001')
        self.used = False

    # tokenizer is not used colbert code base, but is implemented in DocTokenizer
    def tokenize(self, batch_text, add_special_tokens=False):
        raise NotImplementedError()

    # encode is not used colbert code base, but is implemented in DocTokenizer
    def encode(self, batch_text, add_special_tokens=False):
        raise NotImplementedError()

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        batch_text = ['$ ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='longest', truncation='longest_first',
                       return_tensors='pt', max_length=self.doc_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        if self.used is False:
            self.used = True
            print_message("#> Roberta DocTokenizer.tensorize(batch_text[0], batch_background[0], bsize) ==")
            print_message(f"#> Input: {batch_text[0]}, \t\t {bsize}")
            print_message(f"#> Output IDs: {ids[0].size()}, {ids[0]}")
            print_message(f"#> Output Mask: {mask[0].size()}, {mask[0]}")

        if bsize:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask
