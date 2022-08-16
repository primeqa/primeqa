import logging
import torch
import numpy as np
import ujson as json
from primeqa.util.file_utils import jsonl_lines
from primeqa.util.dataloader.distloader_base import MultiFileLoader, DistBatchesBase, sentence_to_inputs
from primeqa.util.transformers_utils.hypers_base import HypersBase
from typing import List
import traceback

logger = logging.getLogger(__name__)


def standard_json_mapper(jobj):
    if 'text_b' in jobj:
        return jobj['id'], jobj['text_a'], jobj['text_b'], jobj['label']
    else:
        return jobj['id'], jobj['text'], jobj['label']


class SeqPairInst:
    __slots__ = 'inst_id', 'toks_a', 'toks_b', 'label', 'teacher_labels'

    def __init__(self, inst_id, toks_a, toks_b, label, teacher_labels):
        self.inst_id = inst_id
        self.toks_a = toks_a
        self.label = label
        self.toks_b = toks_b
        self.teacher_labels = teacher_labels


class SeqPairBatches(DistBatchesBase):
    def __init__(self, insts: List[SeqPairInst], hypers: HypersBase, *, cls_id, sep_id, is_separate, is_single):
        super().__init__(insts, hypers)
        self.hypers = hypers
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.is_separate = is_separate
        self.is_single = is_single

    def make_batch(self, index, insts: List[SeqPairInst]):
        if self.is_single:
            return self.make_batch_single(index, insts)
        elif self.is_separate:
            return self.make_batch_separate(index, insts)
        else:
            return self.make_batch_joined(index, insts)

    def make_batch_single(self, index, insts: List[SeqPairInst]):
        batch_size = len(insts)
        all_lens = np.array([len(i.toks_a) for i in insts], dtype=np.int32)
        maxlen = np.max(all_lens)
        # we make these as numpy first since we can't assign a np.ndarray to torch.tensor
        all_toks = np.zeros((batch_size, maxlen+2), dtype=np.int32)
        all_toks[:, 0] = self.cls_id
        all_token_type = np.zeros((batch_size, maxlen+2), dtype=np.int32)
        all_attention_mask = np.zeros((batch_size, maxlen + 2), dtype=np.int32)
        all_label = np.zeros(batch_size, dtype=np.long)
        ids = [inst.inst_id for inst in insts]
        for i, inst in enumerate(insts):
            offset = 1
            all_toks[i, offset:offset+len(inst.toks_a)] = inst.toks_a
            offset += len(inst.toks_a)
            all_toks[i, offset] = self.sep_id
            offset += 1
            all_attention_mask[i, :offset] = 1
            all_label[i] = inst.label
        tensors = ids, torch.tensor(all_toks, dtype=torch.long), torch.tensor(all_attention_mask, dtype=torch.long), \
                  torch.tensor(all_token_type, dtype=torch.long), torch.tensor(all_label, dtype=torch.long)
        if insts[0].teacher_labels is not None:
            all_teacher_labels = torch.tensor([inst.teacher_labels for inst in insts], dtype=torch.float32)
            tensors = tensors + (all_teacher_labels, )
        return tensors

    def make_batch_joined(self, index, insts: List[SeqPairInst]):
        batch_size = len(insts)
        all_lens = np.array([len(i.toks_a)+len(i.toks_b) for i in insts], dtype=np.int32)
        maxlen = np.max(all_lens)

        # we make these as numpy first since we can't assign a np.ndarray to torch.tensor
        all_toks = np.zeros((batch_size, maxlen+3), dtype=np.int32)
        all_toks[:, 0] = self.cls_id
        all_token_type = np.zeros((batch_size, maxlen+3), dtype=np.int32)
        all_attention_mask = np.zeros((batch_size, maxlen + 3), dtype=np.int32)
        all_label = np.zeros(batch_size, dtype=np.int32)
        ids = [inst.inst_id for inst in insts]
        for i, inst in enumerate(insts):
            # ids and pair_ids are list of ints
            # sequence = tokenizer.build_inputs_with_special_tokens(ids, pair_ids)
            # token_type_ids = tokenizer.create_token_type_ids_from_sequences(ids, pair_ids)
            offset = 1
            all_toks[i, offset:offset+len(inst.toks_a)] = inst.toks_a
            offset += len(inst.toks_a)
            all_toks[i, offset] = self.sep_id
            offset += 1
            seq1_end = offset
            # all_token_type[i, :seq1_end] = 0
            all_toks[i, offset:offset+len(inst.toks_b)] = inst.toks_b
            offset += len(inst.toks_b)
            all_toks[i, offset] = self.sep_id
            offset += 1
            all_token_type[i, seq1_end:offset] = 1
            all_attention_mask[i, :offset] = 1
            all_label[i] = inst.label
        tensors = ids, torch.tensor(all_toks, dtype=torch.long), torch.tensor(all_attention_mask, dtype=torch.long), \
                  torch.tensor(all_token_type, dtype=torch.long), torch.tensor(all_label, dtype=torch.long)
        if insts[0].teacher_labels is not None:
            all_teacher_labels = torch.tensor([inst.teacher_labels for inst in insts], dtype=torch.float32)
            tensors = tensors + (all_teacher_labels, )
        return tensors

    def make_batch_separate(self, index, insts: List[SeqPairInst]):
        def make_seq_tensors(toks: List[np.ndarray]):
            batch_size = len(toks)
            all_lens = np.array([len(t) for t in toks], dtype=np.int32)
            maxlen = np.max(all_lens)
            all_toks = np.zeros((batch_size, maxlen + 2), dtype=np.int32)
            all_toks[:, 0] = self.cls_id
            all_token_type = np.zeros((batch_size, maxlen + 2), dtype=np.int32)
            all_attention_mask = np.zeros((batch_size, maxlen + 2), dtype=np.int32)
            for i, tok in enumerate(toks):
                offset = 1
                all_toks[i, offset:offset + len(tok)] = tok
                offset += len(tok)
                all_toks[i, offset] = self.sep_id
                offset += 1
                all_attention_mask[i, :offset] = 1
            return torch.tensor(all_toks, dtype=torch.long), \
                   torch.tensor(all_attention_mask, dtype=torch.long), \
                   torch.tensor(all_token_type, dtype=torch.long)

        tokens_a, mask_a, token_types_a = make_seq_tensors([i.toks_a for i in insts])
        tokens_b, mask_b, token_types_b = make_seq_tensors([i.toks_b for i in insts])
        tensors = [i.inst_id for i in insts], \
                  tokens_a, mask_a, token_types_a, \
                  tokens_b, mask_b, token_types_b, \
                  torch.tensor([i.label for i in insts], dtype=torch.long)
        if insts[0].teacher_labels is not None:
            all_teacher_labels = torch.tensor([inst.teacher_labels for inst in insts], dtype=torch.float32)
            tensors = tensors + (all_teacher_labels, )
        return tensors


class SeqPairLoader(MultiFileLoader):
    def __init__(self, hypers, per_gpu_batch_size: int, tokenizer, data_dir, *,
                 files_per_dataloader=1, checkpoint_info=None, is_separate=False, is_single=False,
                 json_mapper=standard_json_mapper, teacher_labels=None):
        super().__init__(hypers, per_gpu_batch_size, data_dir,
                         checkpoint_info=checkpoint_info, files_per_dataloader=files_per_dataloader)
        self.tokenizer = tokenizer
        # NOTE: maybe should use tokenizer.cls_token_id, tokenizer.sep_token_id
        self.cls_id, self.sep_id = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
        self.is_separate = is_separate
        self.is_single = is_single
        self.json_mapper = json_mapper
        # just load the entire teacher predictions
        if teacher_labels:
            logger.info(f'loading teacher labels from {teacher_labels}')
            self.id2teacher_labels = dict()
            for line in jsonl_lines(teacher_labels):
                jobj = json.loads(line)
                id = jobj['id']
                preds = jobj['predictions']
                self.id2teacher_labels[id] = np.array(preds, dtype=np.float32)
        else:
            self.id2teacher_labels = None

    def batch_dict(self, batch):
        batch = tuple(t.to(self.hypers.device) for t in batch[1:])
        if len(batch) < 7:
            # single or joined (length should be 4 or 5 depending on presence of teacher_labels)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if self.hypers.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if self.hypers.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            if len(batch) > 3:
                inputs["labels"] = batch[3]
            if len(batch) > 4:
                inputs['teacher_labels'] = batch[4]
        elif len(batch) >= 7:
            # separate (length should be 7 or 8, depending on presence of teacher labels)
            inputs = {"input_ids_a": batch[0], "attention_mask_a": batch[1],
                      "input_ids_b": batch[3], "attention_mask_b": batch[4],
                      "labels": batch[6]}
            if self.hypers.model_type != "distilbert":
                inputs["token_type_ids_a"] = (
                    batch[2] if self.hypers.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                inputs["token_type_ids_b"] = (
                    batch[5] if self.hypers.model_type in ["bert", "xlnet", "albert"] else None
                )
            if len(batch) > 7:
                inputs['teacher_labels'] = batch[7]
        else:
            raise ValueError
        return inputs

    def display_batch(self, batch):
        def to_str(tokens, mask, token_types, bi):
            toks = [str for str in self.tokenizer.convert_ids_to_tokens(tokens[bi])]
            for ti in range(len(toks)):
                toks[ti] = toks[ti] + f'({token_types[bi,ti]},{mask[bi,ti]})'
            return ' '.join(toks)

        def lbl_str(labels, teacher_labels, bi):
            if teacher_labels is None:
                return f'{labels[bi]}'
            else:
                return f'{labels[bi]} ({teacher_labels[bi]})'

        ids = batch[0]
        if len(batch) in [5, 6]:
            tokens, mask, token_types, labels = [t.cpu().numpy() for t in batch[1:5]]
            if len(batch) == 6:
                teacher_labels = batch[5].cpu().numpy()
            else:
                teacher_labels = None
            for bi in range(min(len(ids), 3)):
                logger.info(f'{ids[bi]} is {lbl_str(labels, teacher_labels, bi)}:\n'
                            f'{to_str(tokens, mask, token_types, bi)}')
        else:
            tokens_a, mask_a, token_types_a, tokens_b, mask_b, token_types_b, labels = [t.cpu().numpy() for t in batch[1:8]]
            if len(batch) == 9:
                teacher_labels = batch[8].cpu().numpy()
            else:
                teacher_labels = None
            for bi in range(min(len(ids), 3)):
                logger.info(f'{ids[bi]} is {lbl_str(labels, teacher_labels, bi)}:\n'
                            f'{to_str(tokens_a, mask_a, token_types_a, bi)} || '
                            f'{to_str(tokens_b, mask_b, token_types_b, bi)}')

    def _one_load(self, lines):
        insts = []
        for line in lines:
            jobj = json.loads(line)
            # CONSIDER: do multiprocessing?
            try:
                if self.is_single:
                    inst_id, text_a, label = self.json_mapper(jobj)
                    toks_b = None
                else:
                    inst_id, text_a, text_b, label = self.json_mapper(jobj)
                    toks_b = sentence_to_inputs(text_b, tokenizer=self.tokenizer,
                                                max_seq_length=self.hypers.max_seq_length)
                toks_a = sentence_to_inputs(text_a, tokenizer=self.tokenizer,
                                            max_seq_length=self.hypers.max_seq_length)
                teacher_labels = self.id2teacher_labels[inst_id] if self.id2teacher_labels is not None else None
                sp_inst = SeqPairInst(inst_id, toks_a, toks_b, label, teacher_labels)
                insts.append(sp_inst)
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.warning(f'failed on {line}')

        return SeqPairBatches(insts, self.hypers,
                              cls_id=self.cls_id, sep_id=self.sep_id,
                              is_separate=self.is_separate, is_single=self.is_single)
