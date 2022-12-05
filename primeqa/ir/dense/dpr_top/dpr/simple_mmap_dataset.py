import base64
import ujson as json
import numpy as np
import os
import mmap
import codecs
import threading


def gzip_str(str):
    return codecs.encode(str.encode('utf-8'), 'zlib')
    # return gzip.compress(str.encode('utf-8'))


def gunzip_str(bytes):
    return codecs.decode(bytes, 'zlib').decode('utf-8')
    # return gzip.decompress(bytes).decode('utf-8')


class Corpus:
    def __init__(self, dir):
        # either pass a dir or a specific passagesX.json.gz.records file
        files = []
        base_dir, filename = os.path.split(dir)
        if filename.startswith('passages') and filename.endswith('.json.gz.records'):
            dir = base_dir
            offset_fname = f'offsets{filename[len("passages"):-len(".json.gz.records")]}.npy'
            if not os.path.exists(os.path.join(dir, offset_fname)):
                raise ValueError(f'no offsets file for {filename}!')
            files.append((filename, offset_fname))
        else:
            # for every file like 'passagesX.json.gz.records' there must be a file offsetsX.npy
            for filename in os.listdir(dir):
                if filename.startswith('passages') and filename.endswith('.json.gz.records'):
                    offset_fname = f'offsets{filename[len("passages"):-len(".json.gz.records")]}.npy'
                    if not os.path.exists(os.path.join(dir, offset_fname)):
                        raise ValueError(f'no offsets file for {filename}!')
                    files.append((filename, offset_fname))
        files.sort(key=lambda x: x[0])  # we sort the offsets files, that is our order

        # build offsets table
        # self.offsets will be nx3 self.offsets[i] == file_ndx, start_offset, end_offset
        per_file_offsets = []
        total_passage_count = 0
        for file_pair in files:
            per_file_offsets.append(np.load(os.path.join(dir, file_pair[1])))
            total_passage_count += len(per_file_offsets[-1]) - 1
        self.offsets = np.zeros((total_passage_count, 3), dtype=np.int64)
        total_passage_count = 0
        for file_ndx, file_offsets in enumerate(per_file_offsets):
            passage_count = len(file_offsets)-1
            self.offsets[total_passage_count:total_passage_count + passage_count, 0] = file_ndx
            self.offsets[total_passage_count:total_passage_count + passage_count, 1] = file_offsets[:-1]
            self.offsets[total_passage_count:total_passage_count + passage_count, 2] = file_offsets[1:]
            total_passage_count += passage_count

        # self.mms will be list of memory mapped files (self.files)
        self.mms = []
        self.files = []
        for file_pair in files:
            file = open(os.path.join(dir, file_pair[0]), "r+b")
            self.files.append(file)
            self.mms.append(mmap.mmap(file.fileno(), 0))
        self.pid2ndx = dict()
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if index >= len(self.offsets):
            raise IndexError
        bytes = self.get_raw(index)
        jobj = json.loads(gunzip_str(bytes))
        jobj['vector'] = np.frombuffer(base64.decodebytes(jobj['vector'].encode('ascii')), dtype=np.float16)
        return jobj

    def get_raw(self, index):
        file_ndx, start_offset, end_offset = self.offsets[index]
        return self.mms[file_ndx][start_offset:end_offset]

    def get_by_pid(self, pid):
        with self.lock:
            if len(self.pid2ndx) == 0:
                for ndx in range(len(self.offsets)):
                    obj = json.loads(gunzip_str(self.get_raw(ndx)))
                    pidi = obj['pid']
                    self.pid2ndx[pidi] = ndx
            if pid not in self.pid2ndx:
                return None
            return self[self.pid2ndx[pid]]

    def close(self):
        for mm in self.mms:
            mm.close()
        for file in self.files:
            file.close()
