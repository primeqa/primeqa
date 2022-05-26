import logging
import os
import glob
import gzip
import bz2
import math
import random
import sys
import contextlib
import numpy as np
import base64
import codecs

logger = logging.getLogger(__name__)


def block_shuffle(iter, *, block_size=20000, rand=random):
    """
    shuffle the possibly endless iterator by blocks
    Good shuffling over multiple files: block_shuffle(read_lines(files, shuffled_files=rand), rand=rand, block_size=100000)
    :param iter: the iterator we will yield shuffled items from
    :param block_size: size of memory to use for block shuffling
    :param rand: rand.shuffle will be used on the list block
    :return:
    """
    assert block_size >= 4
    block = []
    for item in iter:
        block.append(item)
        if len(block) >= block_size:
            rand.shuffle(block)
            for _ in range(block_size//2):
                yield block.pop(-1)
    rand.shuffle(block)
    for bi in block:
        yield bi


def jsonl_lines(input_files, completed_files=None, limit=0, report_every=100000, *, errors=None, shuffled=None):
    return read_lines(jsonl_files(input_files, completed_files),
                      limit=limit, report_every=report_every,
                      errors=errors, shuffled_files=shuffled)


def jsonl_files(input_files, completed_files=None):
    return [f for f in expand_files(input_files, '*.jsonl*', completed_files) if not f.endswith('.lock')]


def expand_files(input_files, file_pattern='*', completed_files=None):
    """
    expand the list of files and directories
    :param input_files:
    :param file_pattern: glob pattern for recursive example '*.jsonl*' for jsonl and jsonl.gz
    :param completed_files: these will not be returned in the final list
    :return:
    """
    if type(input_files) is str:
        if ':' in input_files:
            input_files = input_files.split(':')
        else:
            input_files = [input_files]
    # expand input files recursively
    all_input_files = []
    if completed_files is None:
        completed_files = []
    for input_file in input_files:
        if input_file in completed_files:
            continue
        if not os.path.exists(input_file):
            raise ValueError(f'no such file: {input_file}')
        if os.path.isdir(input_file):
            sub_files = glob.glob(input_file + "/**/" + file_pattern, recursive=True)
            sub_files = [f for f in sub_files if not os.path.isdir(f)]
            sub_files = [f for f in sub_files if f not in input_files and f not in completed_files]
            all_input_files.extend(sub_files)
        else:
            all_input_files.append(input_file)
    all_input_files.sort()
    return all_input_files


def read_open(input_file, *, binary=False, errors=None):
    """
    Open text file for reading, assuming compression from extension
    :param input_file:
    :return:
    """
    if binary:
        if input_file.endswith(".gz"):
            return gzip.open(input_file, "rb")
        elif input_file.endswith('.bz2'):
            return bz2.open(input_file, "rb")
        else:
            return open(input_file, "rb")
    else:
        if input_file.endswith(".gz"):
            return gzip.open(input_file, "rt", encoding='utf-8', errors=errors)
        elif input_file.endswith('.bz2'):
            return bz2.open(input_file, "rt", encoding='utf-8', errors=errors)
        else:
            return open(input_file, "r", encoding='utf-8', errors=errors)


def write_open(output_file, *, mkdir=True, binary=False):
    """
    Open text file for writing, assuming compression from extension
    :param output_file:
    :param mkdir:
    :return:
    """
    if mkdir:
        dir = os.path.split(output_file)[0]
        if dir:
            os.makedirs(dir, exist_ok=True)
    if binary:
        if output_file.endswith('.gz'):
            return gzip.open(output_file, 'wb')
        elif output_file.endswith('.bz2'):
            return bz2.open(output_file, 'wb')
        else:
            return open(output_file, 'wb')
    else:
        if output_file.endswith('.gz'):
            return gzip.open(output_file, 'wt', encoding='utf-8')
        elif output_file.endswith('.bz2'):
            return bz2.open(output_file, 'wt', encoding='utf-8')
        else:
            return open(output_file, 'w', encoding='utf-8')


class ShuffledWriter:
    def __init__(self, output_dir, *, extension='.jsonl.gz', num_files=16, rand: random.Random=None):
        self.files = [write_open(os.path.join(output_dir, f'{i}{extension}')) for i in range(num_files)]
        self.rand = rand if rand is not None else random.Random()
        self.current_file = 0
        self.buffer = []
        self.buffer_limit = 1000000

    def write(self, line):
        self.buffer.append(line)
        if len(self.buffer) > self.buffer_limit:
            self.rand.shuffle(self.buffer)
            for _ in range(len(self.buffer)//2):
                self._write(self.buffer.pop(-1))

    def _write(self, line):
        self.files[self.current_file].write(line)
        self.current_file = (self.current_file + 1) % len(self.files)

    def close(self):
        self.rand.shuffle(self.buffer)
        for l in self.buffer:
            self._write(l)
        self.buffer = []
        for f in self.files:
            f.close()


@contextlib.contextmanager
def shuffled_writer(output_dir, *, extension='.jsonl.gz', num_files=16, rand: random.Random=None):
    sw = ShuffledWriter(output_dir, extension=extension, num_files=num_files, rand=rand)
    try:
        yield sw
    finally:
        sw.close()


@contextlib.contextmanager
def stdout_or_file_open(filename=None):
    """
    Opens the file (or stdout if filename is False or '-') for writing.
    Used in 'with' statement.
    :param filename:
    :return:
    """
    if filename and filename != '-':
        fh = write_open(filename)
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()


def np2str(nda, *, dtype=np.float16):
    """
    Convert numpy ndarray to compact string representation
    :param nda: numpy array
    :param dtype: numpy datatype to save the array as
    :return: base64 encoded string of numpy binary
    """
    return base64.b64encode(nda.astype(dtype)).decode('ascii')


def str2np(s: str, *, dtype=np.float16):
    """
    Convert compact string representation of numpy ndarry to numpy vector
    :param s: base64 encoded string of numpy binary
    :param dtype: numpy datatype of the saved array
    :return: 1-D array (shape is not preserved)
    """
    return np.frombuffer(base64.decodebytes(s.encode('ascii')), dtype=dtype)


def gzip_str(str):
    return codecs.encode(str.encode('utf-8'), 'zlib')
    # return gzip.compress(str.encode('utf-8'))


def gunzip_str(bytes):
    return codecs.decode(bytes, 'zlib').decode('utf-8')
    # return gzip.decompress(bytes).decode('utf-8')


def read_lines(input_files, limit=0, report_every=100000, *, errors=None, shuffled_files=None):
    """
    This takes a list (or single) input files and iterates over the lines in them
    :param input_files: Directory name or list of file names
    :param limit: maximum number of lines to read
    :param report_every: log info after this many lines
    :return:
    """
    count = 0
    input_files = expand_files(input_files)
    if shuffled_files:
        if type(shuffled_files) != random.Random:
            shuffled_files = random.Random()
        num_open_blocks = int(math.ceil(len(input_files)/32.0))
        for open_block_i in range(num_open_blocks):
            open_files = [read_open(in_file, errors=errors) for in_file in input_files[open_block_i::num_open_blocks]]
            while len(open_files) > 0:
                fndx = shuffled_files.randint(0, len(open_files)-1)
                next_line = open_files[fndx].readline()
                if next_line:
                    yield next_line
                    count += 1
                    if report_every > 0 and count % report_every == 0:
                        logger.info(f'On line {count}')
                else:
                    open_files[fndx].close()
                    del open_files[fndx]
    else:
        for input_file in input_files:
            with read_open(input_file, errors=errors) as reader:
                for line in reader:
                    yield line
                    count += 1
                    if report_every > 0 and count % report_every == 0:
                        logger.info(f'On line {count} in {input_file}')
                    if 0 < limit <= count:
                        return
