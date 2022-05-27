from oneqa.util.dataloader.distloader_seq_pair import SeqPairLoader
from oneqa.util.transformers_utils.hypers_base import HypersBase
from transformers import AutoTokenizer
from oneqa.util.file_utils import write_open
import os
import ujson as json


class Tester:

    def test_dataloader(self, tmpdir):
        per_gpu_batch_size = 2
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        hypers = HypersBase()
        hypers.max_seq_length = 32
        hypers.num_train_epochs = 1
        data_dir = str(tmpdir)
        for fname in ['0.jsonl', '1.jsonl.bz2', '2.jsonl.gz']:
            with write_open(os.path.join(data_dir, fname)) as f:
                f.write(json.dumps({'id': f'{fname}', 'text_a': 'the cat', 'text_b': 'in the hat', 'label': 0})+'\n')

        for is_separate in [True, False]:
            data = SeqPairLoader(hypers, per_gpu_batch_size, tokenizer, data_dir, is_separate=is_separate)
            loader = data.get_dataloader()
            batch_count = 0
            for batch in loader:
                batch_count += 1
            data.reset()
            for batch in data.all_batches():
                batch_count += 1

        for fname in ['0.jsonl', '1.jsonl.bz2', '2.jsonl.gz']:
            with write_open(os.path.join(data_dir, fname)) as f:
                f.write(json.dumps({'id': f'{fname}', 'text_a': 'the cat', 'label': 0})+'\n')
        data = SeqPairLoader(hypers, per_gpu_batch_size, tokenizer, data_dir, is_single=True)
        loader = data.get_dataloader()
        batch_count = 0
        for batch in loader:
            batch_count += 1
        data.reset()
        for batch in data.all_batches():
            batch_count += 1
