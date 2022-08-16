import torch

from primeqa.ir.dense.colbert_top.colbert.modeling.hf_colbert import HF_ColBERT
from primeqa.ir.dense.colbert_top.colbert.infra import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.modeling.tokenization.utils import _split_into_batches
from primeqa.ir.dense.colbert_top.colbert.utils.utils import batch
from primeqa.ir.dense.colbert_top.colbert.utils.utils import print_message
class QueryTokenizer():
    #     def __init__(self, config: ColBERTConfig):
    def __init__(self, query_maxlen, model_type, attend_to_mask_tokens ):
        # self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(config.checkpoint)
        # assert False

        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(model_type)

        # self.config = config
        # self.query_maxlen = config.query_maxlen
        self.query_maxlen = query_maxlen
        self.background_maxlen = 512 - self.query_maxlen + 1  # FIXME: Make this configurable
        self.attend_to_mask_tokens = attend_to_mask_tokens

        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.convert_tokens_to_ids('[unused0]')
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id

        assert self.Q_marker_token_id == 1 and self.mask_token_id == 103
        self.used = False

    def tokenize(self, batch_text, add_special_tokens=False):
        # assert False

        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst)+3)) for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst)+3)) for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None, context=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [Q] marker
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='max_length', truncation=True,
                       return_tensors='pt', max_length=self.query_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id

        if context is not None:
            assert len(context) == len(batch_text), (len(context), len(batch_text))

            obj_2 = self.tok(context, padding='longest', truncation=True,
                            return_tensors='pt', max_length=self.background_maxlen)

            ids_2, mask_2 = obj_2['input_ids'][:, 1:], obj_2['attention_mask'][:, 1:]  # Skip the first [SEP]

            ids = torch.cat((ids, ids_2), dim=-1)
            mask = torch.cat((mask, mask_2), dim=-1)

        # if self.config.attend_to_mask_tokens:
        if self.attend_to_mask_tokens:
            mask[ids == self.mask_token_id] = 1
            assert mask.sum().item() == mask.size(0) * mask.size(1), mask

        if not self.used:
            self.used = True
            firstbg = (context is None) or context[0]
            print_message("#> BERT QueryTokenizer.tensorize(batch_text[0], batch_background[0], bsize) ==")
            print_message(f"#> Input: {batch_text[0]}, \t\t {firstbg}, \t\t {bsize}")
            print_message(f"#> Output IDs: {ids[0].size()}, {ids[0]}")
            print_message(f"#> Output Mask: {mask[0].size()}, {mask[0]}")

        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        return ids, mask
