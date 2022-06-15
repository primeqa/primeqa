from oneqa.util.file_utils import expand_files, block_shuffle, np2str, str2np, gzip_str, gunzip_str, \
    read_lines, read_open, write_open, ShuffledWriter, shuffled_writer, jsonl_lines
import numpy as np
import os


class Tester:
    def test_expand_files(self):
        assert [] == expand_files([])

    def test_block_shuffle(self):
        orig = list(range(100))
        shuffled = list(block_shuffle(orig, block_size=8))
        shuffled.sort()
        assert orig == shuffled

    def test_np2str(self):
        x = np.random.randn(5).astype(np.float16)
        s = np2str(x)
        x_hat = str2np(s)
        assert all(x == x_hat)

    def test_gzip_str(self):
        assert 'abc' == gunzip_str(gzip_str('abc'))

    def test_read_lines(self):
        assert len(list(read_lines([]))) == 0

    def test_read_write(self, tmpdir):
        with write_open(os.path.join(tmpdir, 'test.txt')) as f:
            f.write('test')
        with read_open(os.path.join(tmpdir, 'test.txt')) as f:
            lines = [line for line in f]
            assert lines == ['test']
        assert ['test'] == list(read_lines(os.path.join(tmpdir, 'test.txt')))

    def test_shuffled_write(self, tmpdir):
        tmpdir = str(tmpdir)
        for ndx, ext in enumerate(['.jsonl', '.jsonl.gz', '.jsonl.bz2']):
            dir = os.path.join(tmpdir, f'{ndx}')
            sw = ShuffledWriter(dir, num_files=16, extension=ext)
            for i in range(100):
                sw.write(f'{i}\n')
            sw.close()
            assert len(expand_files(dir)) == 16
            for line in jsonl_lines(dir):
                assert 0 <= int(line) < 100
            for line in jsonl_lines(dir, shuffled=True):
                assert 0 <= int(line) < 100
