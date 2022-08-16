import torch

from transformers import XLMRobertaTokenizer # there's no Fast version
from primeqa.ir.dense.colbert_top.colbert.modeling.hf_colbert_xlmr import HF_ColBERT_XLMR
from primeqa.ir.dense.colbert_top.colbert.modeling.tokenization.utils import _split_into_batches, _sort_by_length
from primeqa.ir.dense.colbert_top.colbert.utils.utils import print_message

class DocTokenizerXLMR():
    def __init__(self, doc_maxlen, model_type):
        # self.tok = XLMRobertaTokenizer.from_pretrained(model_type)
        self.tok = HF_ColBERT_XLMR.raw_tokenizer_from_pretrained(model_type)

        self.doc_maxlen = doc_maxlen

        self.Q_marker_token, self.D_marker_token_id = '?', 9749  # Hot Beverage
        self.used = False

    # tokenizer is not used colbert code base, but is implemented in DocTokenizer
    def tokenize(self, batch_text, add_special_tokens=False):
        raise NotImplementedError()

    # encode is not used colbert code base, but is implemented in DocTokenizer
    def encode(self, batch_text, add_special_tokens=False):
        raise NotImplementedError()

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [D] marker
        # strangely, prefixing with '. ' introduces _two_ extra tokens [5,6]
        # it seems that 6 is the empty string
        # into the output - I don't understand why ...
        batch_text = ['$ ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='longest', truncation='longest_first',
                       return_tensors='pt', max_length=self.doc_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        if self.used is False:
            self.used = True
            # firstbg = (context is None) or context[0]
            # print()
            print_message("#> XLMR DocTokenizer.tensorize(batch_text[0], batch_background[0], bsize) ==")
            # print(f"#> Input: {batch_text[0]}, \t\t {firstbg}, \t\t {bsize}")
            print_message(f"#> Input: {batch_text[0]}, \t\t {bsize}")
            print_message(f"#> Output IDs: {ids[0].size()}, {ids[0]}")
            print_message(f"#> Output Mask: {mask[0].size()}, {mask[0]}")
            # print()

        if bsize:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask


# In [1]: from colbert.modeling.tokenization import DocTokenizerXLMR

# In [2]: t=DocTokenizerXLMR(50)

# In [3]: t.tensorize(['Here is the answer.', 'Another longer answer is here.'])
# (tensor([[     0,   9749,  11853,     83,     70,  35166,      5,      2,      1],
#          [     0,   9749, 116267,  51713,  35166,     83,   3688,      5,      2]]),
#  tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0],
#          [1, 1, 1, 1, 1, 1, 1, 1, 1]]))


# In [4]:  t.D_marker_token_id
# Out[4]: 9749


# In [5]: t.tok.decode(range(6))
# Out[6]: '<s><pad></s><unk>,.'

