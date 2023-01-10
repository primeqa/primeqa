from primeqa.ir.dense.dpr_top.dataloader.distloader_base import MultiFileLoader, DistBatchesBase
from primeqa.ir.dense.dpr_top.dpr.biencoder_hypers import BiEncoderHypers
from primeqa.ir.dense.dpr_top.util.line_corpus import jsonl_lines
from primeqa.ir.dense.colbert_top.colbert.data.collection import Collection
from primeqa.ir.dense.colbert_top.colbert.data.queries import Queries

import ujson as json
from typing import List
from transformers import PreTrainedTokenizerFast
import torch
import logging
import random
import csv
import re

logger = logging.getLogger(__name__)


class BiEncoderInst:
    __slots__ = 'qry', 'pos_ctx', 'neg_ctx', 'pos_pids', 'ctx_pids'

    def __init__(self, qry, pos_ctx, neg_ctx, pos_pids, ctx_pids):
        self.qry = qry
        self.pos_ctx = pos_ctx
        self.neg_ctx = neg_ctx
        self.pos_pids = pos_pids
        self.ctx_pids = ctx_pids
        assert len(ctx_pids) == 2  # TODO: not in the DPR-style data


class BiEncoderBatches(DistBatchesBase):
    def __init__(self, insts: List[BiEncoderInst], hypers: BiEncoderHypers,
                 qry_tokenizer: PreTrainedTokenizerFast, ctx_tokenizer: PreTrainedTokenizerFast):
        super().__init__(insts, hypers)
        self.qry_tokenizer = qry_tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.hypers = hypers
        self.batched_instances = []

    def create_conflict_free_batches(self, random):
        """
        Since we do batch negatives we want to ensure that the batch does not contain instances
        where the batch negatives contain positives.
        :return:
        """
        pushed_to_leftover = 0
        batch_neg_pids = set()  # the pids that our batch will call batch negatives (for any instance we might add to the batch)
        batch_pos_pids = set()  # the actual positives across all instances in our batch
        leftover_insts = []
        current_batch = []
        current_batch_leftover = []
        if self.hypers.training_data_type != 'kgi_jsonl' and self.hypers.force_confict_free_batches:
            raise NotImplementedError(f"Confict free batches for {self.hypers.training_data_type} data are not implemented (yet).")

        while len(self.insts) + len(leftover_insts) >= self.batch_size:
            # grab an instance
            if len(leftover_insts) > 0:
                inst = leftover_insts.pop()
            else:
                inst = self.insts.pop()
            # adding it to our batch should not violate our hard negative constraint:
            #  no positive or hard negative for one instance should be a positive for another instance
            if not self.hypers.force_confict_free_batches or \
                (all([pp not in batch_neg_pids for pp in inst.pos_pids]) and
                    all([np not in batch_pos_pids for np in inst.ctx_pids])):
                current_batch.append(inst)
                for cp in inst.ctx_pids:
                    batch_neg_pids.add(cp)
                for pp in inst.pos_pids:
                    batch_pos_pids.add(pp)
            else:
                current_batch_leftover.append(inst)  # this instance can't go in the current batch
                pushed_to_leftover += 1
            if len(current_batch) == self.batch_size:
                self.batched_instances.append(current_batch)
                leftover_insts.extend(current_batch_leftover)
                random.shuffle(leftover_insts)
                current_batch_leftover = []
                current_batch = []
                batch_neg_pids = set()
                batch_pos_pids = set()

        logger.warning(f'out of {len(self.batched_instances)} batches of size {self.batch_size}, '
                       f'pushed {pushed_to_leftover} out of batch due to conflict, '
                       f'{len(current_batch_leftover)} pushed out completely')
        if len(current_batch_leftover) > 2 * self.batch_size:
            logger.error(f'So many can not be batched! {len(current_batch_leftover)} unbatched!')
        self.insts = None  # no longer use insts, only batched_instances

    def post_init(self, *, batch_size, displayer=None, uneven_batches=False, random=None):
        self.batch_size = batch_size
        assert not uneven_batches
        assert random is not None
        random.shuffle(self.insts)
        self.create_conflict_free_batches(random) # this is why we override post_init and __getitem__
        self.num_batches = len(self.batched_instances)
        self.displayer = displayer
        self.uneven_batches = uneven_batches
        if self.hypers.world_size != 1:
            self._distributed_min()

    def __getitem__(self, index):
        if index >= self.num_batches:
            raise IndexError
        batch_insts = self.batched_instances[index]
        batch = self.make_batch(index, batch_insts)
        if index == 0 and self.displayer is not None:
            self.displayer(batch)
        return batch

    def make_batch(self, index, insts: List[BiEncoderInst]):
        ctx_titles = [title for i in insts for title in [i.pos_ctx[0], i.neg_ctx[0]]]
        ctx_texts = [text for i in insts for text in [i.pos_ctx[1], i.neg_ctx[1]]]
        # if index == 0:
        #     logger.info(f'titles = {ctx_titles}\ntexts = {ctx_texts}')
        qrys = [i.qry for i in insts]
        ctxs_tensors = self.ctx_tokenizer(ctx_titles, ctx_texts, max_length=self.hypers.seq_len_c,
                                          truncation=True, padding="longest", return_tensors="pt")
        if type(qrys[0]) == str:
            qrys_tensors = self.qry_tokenizer(qrys, max_length=self.hypers.seq_len_q,
                                              truncation=True, padding="longest", return_tensors="pt")
        elif type(qrys[0]) == dict:
            qrys_tensors = self.qry_tokenizer([q['title'] for q in qrys], [q['text'] for q in qrys],
                                              max_length=self.hypers.seq_len_q,
                                              truncation=True, padding="longest", return_tensors="pt")
        else:
            raise ValueError
        positive_indices = torch.arange(len(insts), dtype=torch.long) * 2
        assert qrys_tensors['input_ids'].shape[0] * 2 == ctxs_tensors['input_ids'].shape[0]
        return qrys_tensors['input_ids'], qrys_tensors['attention_mask'], \
               ctxs_tensors['input_ids'], ctxs_tensors['attention_mask'], \
               positive_indices


class BiEncoderLoader(MultiFileLoader):
    def __init__(self, hypers: BiEncoderHypers, per_gpu_batch_size: int, qry_tokenizer, ctx_tokenizer, data_dir,
                 positive_pid_file, *, files_per_dataloader=1, checkpoint_info=None):
        super().__init__(hypers, per_gpu_batch_size, data_dir,
                         checkpoint_info=checkpoint_info, files_per_dataloader=files_per_dataloader)
        self.qry_tokenizer = qry_tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.id2pos_pids = dict()
        if self.hypers.training_data_type == 'kgi_jsonl':
            for line in jsonl_lines(positive_pid_file, file_suffix='*.jsonl*'):
                jobj = json.loads(line)
                self.id2pos_pids[jobj['id']] = jobj['positive_pids']
        if self.hypers.training_data_type == 'num_triples':
            self.queries = Queries.cast(self.hypers.queries)
            self.collection = Collection.cast(self.hypers.collection)

    def batch_dict(self, batch):
        """
        :param batch: input_ids_q, attention_mask_q, input_ids_c, attention_mask_c, positive_indices
        :return:
        """
        batch = tuple(t.to(self.hypers.device) for t in batch)
        return {'input_ids_q': batch[0], 'attention_mask_q': batch[1],
                'input_ids_c': batch[2], 'attention_mask_c': batch[3],
                'positive_indices': batch[4]}

    def display_batch(self, batch):
        input_ids_q = batch[0]
        input_ids_c = batch[2]
        positive_indices = batch[4]
        logger.info(f'{input_ids_q.shape} queries and {input_ids_c.shape} contexts\n{positive_indices}')
        qndx = random.randint(0, input_ids_q.shape[0]-1)
        logger.info(f'   query: {self.qry_tokenizer.decode(input_ids_q[qndx])}')
        logger.info(f'   query: {input_ids_q[qndx]}')
        logger.info(f'positive: {self.ctx_tokenizer.decode(input_ids_c[positive_indices[qndx]])}')
        logger.info(f'positive: {input_ids_c[positive_indices[qndx]]}')
        logger.info(f'negative: {self.ctx_tokenizer.decode(input_ids_c[1+positive_indices[qndx]])}')
        logger.info(f'negative: {input_ids_c[1+positive_indices[qndx]]}')

    def _one_load(self, lines):
        insts = []

        if self.hypers.training_data_type == 'dpr':
            for line in lines:
                # "line" is a dictionary here
                qry = line['question']
                # using all positives, one negative per positive, hard negative if available
                for positive_ctx in line['positive_ctxs']:
                    positive = positive_ctx['title'], positive_ctx['text']

                    for negs, max_negs_num in [(line['negative_ctxs'], self.hypers.max_negatives), (line['hard_negative_ctxs'], self.hypers.max_hard_negatives)]:
                        for pos in range(min(len(negs), max_negs_num)):
                            neg_ndx = random.randint(0, min(len(negs), self.hypers.sample_negative_from_top_k)-1)
                            neg = negs[neg_ndx]['title'], negs[neg_ndx]['text']
                            ctx_pids = [-1, -1]  # TODO: just a hack to avoid the assert in BiEncoderInst.__init__
                            pos_pids = []
                            assert len(positive) == 2
                            assert len(neg) == 2
                            insts.append(BiEncoderInst(qry, positive, neg, pos_pids, ctx_pids))
        elif self.hypers.training_data_type == 'kgi_jsonl':
            for line in lines:
                jobj = json.loads(line)
                qry = jobj['query']
                positive = jobj['positive']['title'], jobj['positive']['text']
                negs = jobj['negatives']
                if len(negs) == 0:
                    logger.warning(f'bad instance! {len(negs)} negatives')
                    continue
                neg_ndx = random.randint(0, min(len(negs), self.hypers.sample_negative_from_top_k)-1)
                hard_neg = negs[neg_ndx]['title'], negs[neg_ndx]['text']
                ctx_pids = [jobj['positive']['pid'], negs[neg_ndx]['pid']]
                pos_pids = self.id2pos_pids[jobj['id']]
                assert len(positive) == 2
                assert len(hard_neg) == 2
                insts.append(BiEncoderInst(qry, positive, hard_neg, pos_pids, ctx_pids))
        elif self.hypers.training_data_type == 'text_triples':
            for line in lines:
                [qry, positive_text, negative_text] = next(csv.reader([line], delimiter="\t", quotechar='"'))   # TODO: is this slow?
                ctx_pids = [-1, -1]  # TODO: just a hack to avoid the assert in BiEncoderInst.__init__
                pos_pids = []
                positive = '', positive_text  # NOTE: the titles are not delimited in the .tsv files
                hard_neg = '', negative_text
                insts.append(BiEncoderInst(qry, positive, hard_neg, pos_pids, ctx_pids))
        elif self.hypers.training_data_type == 'text_triples_with_title':
            for line in lines:
                [qry, positive, negative] = next(csv.reader([line], delimiter="\t", quotechar='"'))   # TODO: is this slow?
                positive_title, positive_text = re.split(r" \| ", positive, maxsplit=1)
                negative_title, negative_text = re.split(r" \| ", negative, maxsplit=1)
                ctx_pids = [-1, -1]  # TODO: just a hack to avoid the assert in BiEncoderInst.__init__
                pos_pids = []

                # to deal with data based on quoted tsv
                [positive_title] = next(csv.reader([positive_title], delimiter="\n", quotechar='"'))
                [positive_text] = next(csv.reader([positive_text], delimiter="\n", quotechar='"'))
                [negative_title] = next(csv.reader([negative_title], delimiter="\n", quotechar='"'))
                [negative_text] = next(csv.reader([negative_text], delimiter="\n", quotechar='"'))

                positive = positive_title, positive_text  # NOTE: the titles are not delimited in the .tsv files
                hard_neg = negative_title, negative_text
                insts.append(BiEncoderInst(qry, positive, hard_neg, pos_pids, ctx_pids))
        elif self.hypers.training_data_type == 'num_triples':
            for line in lines:
                [query_id, positive_id, negative_id] = json.loads(line)
                qry = self.queries[str(query_id)]
                # because of title and text handling in colbert.evaluation.loaders.load_collection
                positive_title, positive_text = re.split(r" \| ", self.collection[positive_id], maxsplit=1)
                negative_title, negative_text = re.split(r" \| ", self.collection[negative_id], maxsplit=1)

                # to deal with data based on quoted tsv
                [positive_title] = next(csv.reader([positive_title], delimiter="\n", quotechar='"'))
                [positive_text] = next(csv.reader([positive_text], delimiter="\n", quotechar='"'))
                [negative_title] = next(csv.reader([negative_title], delimiter="\n", quotechar='"'))
                [negative_text] = next(csv.reader([negative_text], delimiter="\n", quotechar='"'))

                ctx_pids = [-1, -1]  # TODO: just a hack to avoid the assert in BiEncoderInst.__init__
                pos_pids = []
                positive = positive_title, positive_text  # NOTE: the titles are not delimited in the .tsv files
                hard_neg = negative_title, negative_text
                insts.append(BiEncoderInst(qry, positive, hard_neg, pos_pids, ctx_pids))
        return BiEncoderBatches(insts, self.hypers, self.qry_tokenizer, self.ctx_tokenizer)
