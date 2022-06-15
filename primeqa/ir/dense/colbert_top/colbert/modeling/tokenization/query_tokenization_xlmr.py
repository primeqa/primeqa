import torch

from primeqa.ir.dense.colbert_top.colbert.modeling.hf_colbert_xlmr import HF_ColBERT_XLMR
from transformers import XLMRobertaTokenizer # there's no Fast version
from primeqa.ir.dense.colbert_top.colbert.modeling.tokenization.utils import _split_into_batches
from primeqa.ir.dense.colbert_top.colbert.utils.utils import print_message

# only the following official escape sequences are available
# 0            0     <s>
# 1            1   <pad>
# 2            2    </s>
# 3            3   <unk>
# 250001  250001  <mask>
#
# we will use the following unofficial escape sequences:
# 246260,246260,?,9748 '\u2614' Umbrella with Rain Drops
# 245281,245281,?,9749 '\u2615' Hot Beverage


class QueryTokenizerXLMR():
    def __init__(self, query_maxlen, model_type):
        # self.tok = XLMRobertaTokenizer.from_pretrained(model_type)
        self.tok = HF_ColBERT_XLMR.raw_tokenizer_from_pretrained(model_type)
        self.query_maxlen = query_maxlen

        self.Q_marker_token, self.Q_marker_token_id = '?', 9748  # Umbrellawith Rain Drops
        self.mask_token, self.mask_token_id = self.tok.pad_token, self.tok.pad_token_id

#        assert self.Q_marker_token_id == 1 and self.mask_token_id == 103
        self.used = False

    # tokenizer is not used colbert code base, but is implemented in QueryTokenizer
    def tokenize(self, batch_text, add_special_tokens=False):
        raise NotImplementedError()

    # encode is not used colbert code base, but is implemented in QueryTokenizer
    def encode(self, batch_text, add_special_tokens=False):
        raise NotImplementedError()

    def tensorize(self, batch_text, bsize=None, context=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [Q] marker
        # strangely, prefixing with '. ' introduces _two_ extra tokens [5,6]
        # it seems that 6 is the empty string
        # into the output - I don't understand why ...
        batch_text = ['$ ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='max_length', truncation=True,
                       return_tensors='pt', max_length=self.query_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']
        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        #
        # roberta tokenizer has pad_token_id=1, <s>=0, so the following statement must be omitted
        #        ids[ids == 0] = self.mask_token_id
        # I'm keeping commented-out code here in case of comparison with QueryTokenizer.py (bert)

        if context is not None:
            print_message(f"#> length of context: {len(context)}")

        if not self.used:
            self.used = True
            firstbg = (context is None) or context[0]

            print_message("#> XMLR QueryTokenizer.tensorize(batch_text[0], batch_background[0], bsize) ==")
            print_message(f"#> Input: {batch_text[0]}, \t\t {firstbg}, \t\t {bsize}")
            print_message(f"#> Output IDs: {ids[0].size()}, {ids[0]}")
            print_message(f"#> Output Mask: {mask[0].size()}, {mask[0]}")

        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        return ids, mask



# In [1]: from colbert.modeling.tokenization import QueryTokenizerXLMR

# In [2]: t=QueryTokenizerXLMR(50)

# In [3]: t.tensorize(['what is the answer?', 'is that not completely ridiculously false?'])
# (tensor([[     0,   9748,   2367,     83,     70,  35166,     32,      2,      1,
#                1,      1,      1,      1,      1,      1,      1,      1,      1,
#                1,      1,      1,      1,      1,      1,      1,      1,      1,
#                1,      1,      1,      1,      1,      1,      1,      1,      1,
#                1,      1,      1,      1,      1,      1,      1,      1,      1,
#                1,      1,      1,      1,      1],
#          [     0,   9748,     83,    450,    959,  64557, 236873,    538,  98320,
#               32,      2,      1,      1,      1,      1,      1,      1,      1,
#                1,      1,      1,      1,      1,      1,      1,      1,      1,
#                1,      1,      1,      1,      1,      1,      1,      1,      1,
#                1,      1,      1,      1,      1,      1,      1,      1,      1,
#                1,      1,      1,      1,      1]]),
#  tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0],
#          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0]]))

# In [4]: t.Q_marker_token_id
# Out[4]: 9748

# In [5]: t.mask_token_id
# Out[6]: 1

# In [6]: t.tok.decode(range(5))
# Out[7]: '<s><pad></s><unk>,'

