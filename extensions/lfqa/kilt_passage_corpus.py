from primeqa.ir.dense.dpr_top.util.line_corpus import read_lines, write_open
from primeqa.ir.dense.dpr_top.util.args_help import fill_from_args
import ujson as json
import csv
import os
import re
import logging
import itertools

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


class Options:
    def __init__(self):
        self.kilt_corpus = ''
        self.output_dir = ''
        self.passage_ids = ''
        self.num_output_files = 1
        self.max_passage_words = 100
        self.min_passage_words = 50
        self.__required_args__ = ['kilt_corpus', 'output_dir', 'passage_ids']


_WHITESPACE = re.compile(r'\s+')
def clean_text(paragraph: str):
    # handle:
    #  Section::::
    #  BULLET::::-
    return _WHITESPACE.sub(' ', paragraph.replace('::::', ': ')).strip()


opts = Options()
fill_from_args(opts)

out_files = [write_open(os.path.join(opts.output_dir, f'kilt_knowledgesource_{i}.tsv')) for i in range(opts.num_output_files)]
tsv_writers = [csv.writer(f, delimiter="\t") for f in out_files]
for tsv_writer in tsv_writers:
    tsv_writer.writerow(['id', 'text', 'title'])
passage_id_file = write_open(opts.passage_ids)
# passage id format is: doc_id::[start_para,end_para]
# if the interval is doc_id::[start_para,end_para) then some (but not all) of end_para is included
# if the interval is doc_id::(start_para,end_para] then some (but not all) of start_para is included
passage_count = 0


def find_para_range(word_counts, start_para):
    # find the end_para that fills up our passage
    end_para = start_para+1
    while end_para < len(word_counts) and sum(word_counts[start_para:end_para]) < opts.max_passage_words:
        end_para += 1
    if sum(word_counts[start_para:end_para]) > opts.max_passage_words and sum(word_counts[start_para:end_para-1]) >= opts.min_passage_words:
        # don't need the last paragraph
        end_para -= 1
    if end_para == len(paragraphs):
        # include some earlier paragraphs
        while start_para > 0 and sum(word_counts[start_para:end_para]) < opts.min_passage_words:
            start_para -= 1
    return start_para, end_para


def write(doc_id, title, paragraphs, word_counts, start_para):
    """
    write a passage starting at start_para
    :param doc_id:
    :param title:
    :param paragraphs:
    :param word_counts:
    :param start_para:
    :return: the next start_para for a passage
    """
    global passage_count
    assert len(paragraphs) == len(word_counts)
    assert 0 <= start_para < len(paragraphs)

    orig_start_para = start_para
    start_para, end_para = find_para_range(word_counts, start_para)
    full_end_para = end_para

    # enough words or all the words
    words = list(itertools.chain(*paragraphs[start_para:end_para]))
    assert len(words) >= opts.min_passage_words or (start_para == 0 and end_para == len(paragraphs))

    if opts.min_passage_words <= len(words) <= opts.max_passage_words:
        # our paragraph boundaries work out
        passage_id = f'{doc_id}::[{start_para},{end_para-1}]'
    elif len(words) > opts.max_passage_words:
        # we need to truncate some of the first or last paragraph
        if start_para == orig_start_para:
            # chop from the end
            assert start_para == end_para+1 or sum(word_counts[start_para:end_para-1]) < opts.min_passage_words
            words = words[:opts.max_passage_words]
            passage_id = f'{doc_id}::[{start_para},{end_para-1})'
            full_end_para = end_para-1
        else:
            # chop from the begining
            assert start_para == end_para+1 or sum(word_counts[start_para+1:end_para]) < opts.min_passage_words
            words = words[-opts.max_passage_words:]
            passage_id = f'{doc_id}::({start_para},{end_para-1}]'
    else:
        # the document is too short, we take it all as a single too-short passage
        assert len(words) < opts.min_passage_words and end_para-start_para == len(paragraphs)
        passage_id = f'{doc_id}::[{start_para},{end_para-1}]'
    assert len(words) <= opts.max_passage_words
    text = ' '.join(words)
    # out_files[passage_count % len(out_files)].write(
    #     json.dumps({'pid': passage_id, 'title': title, 'text': text}) + '\n')
    tsv_writers[passage_count % len(out_files)].writerow([(passage_count+1),text,title])
    passage_id_file.write(f'{passage_id}\n')
    passage_count += 1
    return max(orig_start_para+1, full_end_para)


too_short_document_count = 0
too_long_paragraph_count = 0
total_paragraphs = 0
for line in read_lines(opts.kilt_corpus):
    jobj = json.loads(line)
    doc_id = jobj['wikipedia_id']
    title = jobj['wikipedia_title']
    paragraphs = [clean_text(p).split(' ') for p in jobj['text']]
    word_counts = [len(p) for p in paragraphs]
    total_paragraphs += len(paragraphs)
    too_long_paragraph_count += sum([wc > opts.max_passage_words for wc in word_counts])
    too_short_document_count += 1 if sum(word_counts) < opts.min_passage_words else 0
    start_para = 0
    while start_para < len(paragraphs):
        start_para = write(doc_id, title, paragraphs, word_counts, start_para)

print(f'{total_paragraphs} paragraphs, too long {too_long_paragraph_count}, docs too short {too_short_document_count}')


for out_file in out_files:
    out_file.close()
passage_id_file.close()
