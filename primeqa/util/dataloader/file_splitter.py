import logging
import os
import random
import numpy as np
from primeqa.util.file_utils import jsonl_lines, write_open
from primeqa.util.args_helper import fill_from_args
import ujson as json

logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        self.input = ''
        self.outdirs = ''
        self.split_fractions = '1.0'
        self.extension = '.jsonl.gz'
        self.file_counts = '4'
        self.id_field = ''  # if set the splits will be by id
        self.exclude = ''  # the ids present here will be excluded
        self.__required_args__ = ['input', 'outdirs']


def main(opts: Options):
    if opts.exclude and not opts.id_field:
        opts.id_field = 'id'
    # sort out output
    outdirs = [d.strip() for d in opts.outdirs.split(',')]
    splits = np.array([float(s) for s in opts.split_fractions.split(',')], dtype=np.float32)
    assert len(outdirs) == len(splits)
    splits /= splits.sum()
    counts = np.zeros(len(outdirs), dtype=np.int32)
    logger.info(f'Splits:')
    for out, frac in zip(outdirs, splits):
        logger.info(f'  {frac}: {out}')
    file_counts = [int(fc) for fc in opts.file_counts.split(',')]
    if len(file_counts) == 1:
        file_counts = file_counts * len(outdirs)
    assert len(file_counts) == len(outdirs)
    outfiles = [[None] * fc for fc in file_counts]

    # open all files
    for ofi, outdir in zip(outfiles, outdirs):
        for j in range(len(ofi)):
            ofi[j] = write_open(os.path.join(outdir, f'{j}{opts.extension}'))

    exclude_ids = set()
    if opts.exclude:
        for line in jsonl_lines(opts.exclude):
            try:
                inst_id = json.loads(line)[opts.id_field]
                exclude_ids.add(inst_id)
            except:
                logger.warning(f'bad line: {line}')
    excluded_count = 0

    # split lines between files
    for line in jsonl_lines(opts.input):
        if opts.id_field:
            inst_id = json.loads(line)[opts.id_field]
            if inst_id in exclude_ids:
                excluded_count += 1
                continue
            insplit_point = random.Random(inst_id).random()
        else:
            insplit_point = random.random()
        cummulative = 0.0
        outdir_ndx = splits.shape[0]-1
        for i in range(splits.shape[0]-1):
            cummulative += splits[i]
            if insplit_point <= cummulative:
                outdir_ndx = i
                break
        ofi = outfiles[outdir_ndx]
        ofi[counts[outdir_ndx] % len(ofi)].write(line)
        counts[outdir_ndx] += 1

    if opts.exclude:
        logger.info(f'Set to exclude {len(exclude_ids)}, excluded {excluded_count}')

    # close all files
    for i, ofi in enumerate(outfiles):
        print(f'wrote {counts[i]} in {outdirs[i]}')
        for j in range(len(ofi)):
            ofi[j].close()


"""
python dataloader/file_splitter.py  --outdirs train,dev --split_fractions 0.8,0.2 --input 00.jsonl.gz
"""

if __name__ == "__main__":
    opts = Options()
    fill_from_args(opts)
    main(opts)
