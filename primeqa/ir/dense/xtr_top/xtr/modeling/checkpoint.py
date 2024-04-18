import os
import json
import torch

from tqdm import tqdm
from transformers import AutoConfig

from primeqa.ir.dense.xtr_top.xtr.modeling.XTR import XTR
from primeqa.ir.dense.xtr_top.xtr.tokenization.custom_tokenization import XTRTokenizer

from transformers import AutoTokenizer, AutoModel


class Checkpoint:
    def __init__(self, args):

        self.tokenizer = XTRTokenizer(args)
        
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.config.dim = args.dim
        self.model = XTR.from_pretrained(args.model_name_or_path, config=self.config)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.docFromText_used = False

    def query(self, input_ids, attention_mask):
        with torch.no_grad():
            mask = attention_mask.to(device=self.model.device)
            input_ids = input_ids.to(device=self.model.device)
            Q = self.model.encode(input_ids.to(device=self.model.device), mask) 
            
            Q[input_ids == 1] = (Q * mask.unsqueeze(-1)).mean(1)
            Q = Q * mask.unsqueeze(2)
            #Q = torch.nn.functional.normalize(Q, p=2, dim=2)
            if Q.device.type == "cuda":
                Q = Q.half()
            return Q, mask

    def doc(self, input_ids, attention_mask, keep_dims=True):
        attention_mask = attention_mask.to(device=self.model.device)
        input_ids = input_ids.to(device=self.model.device)
        with torch.no_grad():
            D = self.model.encode(input_ids, attention_mask) 
            D = D * attention_mask.unsqueeze(2)
            D[input_ids == 1] = D.mean(1)
            #D = torch.nn.functional.normalize(D, p=2, dim=2)
            if D.device.type == "cuda":
                D = D.half()
            return D, attention_mask.bool()

    def queryFromText(self, queries, bsize=None):
        if bsize:
            batches = self.tokenizer.tensorize(queries, bsize=bsize, text_type='query')
            batches = [self.query(input_ids, attention_mask) for input_ids, attention_mask in batches]
            return torch.cat(batches)

        encodings = self.tokenizer.tensorize(queries, text_type='query')
        return self.query(encodings['input_ids'], encodings['attention_mask'])

    def docFromText(self, docs, bsize=None, keep_dims=True, showprogress=False, return_tokens=False):
        assert keep_dims in [True, False, 'flatten']
        if not self.docFromText_used:
            print(f"#> checkpoint, docFromText, Input: {docs[0]}, \t\t {bsize}")

        if bsize:
            text_batches, reverse_indices = self.tokenizer.tensorize(docs, bsize=bsize, text_type='passage')

            if not self.docFromText_used:
                print(f"#> checkpoint, docFromText, Output IDs: {text_batches[0]}")
                self.docFromText_used = True

            returned_text = []
            if return_tokens:
                returned_text = [text for batch in text_batches for text in batch[0]]
                returned_text = [returned_text[idx] for idx in reverse_indices.tolist()]
                returned_text = [returned_text]

            keep_dims_ = 'return_mask' if keep_dims == 'flatten' else keep_dims
            batches = [self.doc(input_ids, attention_mask, keep_dims=keep_dims_)
                       for input_ids, attention_mask in tqdm(text_batches, disable=not showprogress)]

            if keep_dims is True:
                D = _stack_3D_tensors(batches)
                return (D[reverse_indices], *returned_text)

            elif keep_dims == 'flatten':
                D, mask = [], []

                for D_, mask_ in batches:
                    D.append(D_)
                    mask.append(mask_)

                D, mask = torch.cat(D)[reverse_indices], torch.cat(mask)[reverse_indices]

                doclens = mask.squeeze(-1).sum(-1).tolist()

                D = D.view(-1, self.config.dim)
                D = D[mask.bool().flatten()].cpu()

                return (D, doclens, *returned_text)

            assert keep_dims is False

            D = [d for batch in batches for d in batch]
            return ([D[idx] for idx in reverse_indices.tolist()], *returned_text)

        input_ids, attention_mask = self.tokenizer.tensorize(docs, text_type='passage')
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims)

def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output
