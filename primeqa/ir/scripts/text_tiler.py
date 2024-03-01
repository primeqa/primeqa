import unicodedata
import re
from typing import List, Any, Dict
import pyizumo

class TextTiler:
    """
    TextTiler is a class that helps splits a long string into segments that are
    of a specified length (in LLM tokenizer units), and can follow sentence boundaries
    or not.
    """
    def __init__(self, max_doc_size, stride, tokenizer,
                 aligned_on_sentences: bool = True):
        self.max_doc_size = max_doc_size
        self.stride = stride
        self.tokenizer = tokenizer
        self.product_counts = {}
        self.aligned_on_sentences = aligned_on_sentences
        self.nlp = None

    @staticmethod


    def create_tiles(self,
                     id_:str,
                     title,
                     text,
                     max_doc_size=None,
                     stride=None,
                     remove_url=True,
                     normalize_text=False,
                     title_handling:str="all",
                     template:Dict[str: str]= {},
                     ):
        """
        Converts a given document or passage (from 'output.json') to a dictionary, splitting the text as necessary.
        :param id_: str - the prefix of the id of the resulting piece/pieces
        :param title: str - the title of the new piece
        :param text: the input text to be split
        :param max_doc_size: int - the maximum size (in word pieces) of the resulting sub-document/sub-passage texts
        :param stride: int - the stride/overlap for consecutive pieces
        :param remove_url: Boolean - if true, URL in the input text will be replaced with "URL"
        :param normalize_text: boolean - if true, normalize the text for UTF-8 encoding.
        :param title_handling:str - can be ['all', 'first', 'none'] - defines how the title will be added to the text pieces:
           * 'all': the title is added to each of the created tiles
           * 'first': the title is added to only the first tile
           * 'none': the title is not added to any of the resulting tiles
        :param template: Dict[str: str] - the template for each json entry; each entry will be appended an 'id' and 'text'
                         keys with the split text.
        :return - a list of indexable items, each containing a title, id, text, and url.
        """
        if max_doc_size is None:
            max_doc_size = self.max_doc_size
        if stride is None:
            stride = self.stride
        itm = template
        pieces = []
        url = r'https?://(?:www\.)?(?:[-a-zA-Z0-9@:%._\+~#=]{1,256})\.(:?[a-zA-Z0-9()]{1,6})(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)*\b'
        if text.find("With this app") >= 0 or text.find("App ID") >= 0:
            itm['app_name'] = title
        if not text.startswith(title):
            expanded_text = f"{title}\n{text}"
        else:
            expanded_text = text

        if remove_url:
            # The normalization below deals with some issue in the re library - it would get stuck
            # if the URL has some strange chars, like `\xa0`.
            if normalize_text:
                text = re.sub(url, 'URL', unicodedata.normalize("NFKC", text))
            else:
                text = re.sub(url, 'URL', text)

        if self.tokenizer is not None:
            merged_length = self.get_tokenized_length(text=expanded_text)
            if merged_length <= max_doc_size:
                itm.update({'id': f"{id_}-0-{len(text)}", 'text': expanded_text})
                pieces.append(itm.copy())
            else:
                maxl = max_doc_size  # - title_len
                psgs, inds = self.split_text(text=text, max_length=maxl, title=title,
                                             stride=stride, tokenizer=self.tokenizer, title_handling=title_handling)
                for pi, (p, index) in enumerate(zip(psgs, inds)):
                    itm.update({
                        'id': f"{id_}-{index[0]}-{index[1]}",
                        'text': (f"{title}\n{p}"
                                 if title_handling=="any" or title_handling=="first" and pi==0
                                 else p)
                    })
                    pieces.append(itm.copy())
        else:
            itm.update({'id': id_, 'text': expanded_text})
            pieces.append(itm.copy())
        return pieces

    def get_tokenized_length(self, text):
        """
        Returns the size of the <text> after being tokenized by <tokenizer>
        :param text: str - the input text
        :return the length (in word pieces) of the tokenized text.
        """
        if self.tokenizer is not None:
            toks = self.tokenizer(text)
            return len(toks['input_ids'])
        else:
            return -1

    def split_text(self, text: str, tokenizer, title: str = "", max_length: int = -1, stride: int = -1,
                   language_code='en', title_handling:str='any') \
            -> tuple[list[str], list[list[int | Any]]]:
        """
        Method to split a text into pieces that are of a specified <max_length> length, with the
        <stride> overlap, using an HF tokenizer.
        :param text: str - the text to split
        :param tokenizer: HF Tokenizer
           - the tokenizer to do the work of splitting the words into word pieces
        :param title: str - the title of the document
        :param max_length: int - the maximum length of the resulting sequence
        :param stride: int - the overlap between windows
        :param language_code: str - the 2-letter language code of the text (default, 'en').
        :param title_handling: str - can be ['all', 'first', 'none'] - defines how the title will be added to the text pieces:
           * 'all': the title is added to each of the created tiles
           * 'first': the title is added to only the first tile
           * 'none': the title is not added to any of the resulting tiles
        :return: Tuple[list[str], list[list[int | Any]]] - returns a pair of tile string list and a position list (each entry
                 is a list of the start/end position of the corresponding tile in the first returned argument).
        """
        def get_expanded_text(text, pos=0, title_handling="all"):
            if title_handling == "none" or title_handling == "first" and pos>0:
                return title
            else:
                return f"{title}\n{text}"

        if max_length == -1:
            max_length = self.max_doc_size
        if stride == -1:
            stride = self.stride
        text = re.sub(r' {2,}', ' ', text, flags=re.MULTILINE)  # remove multiple spaces.
        if max_length is not None:
            tok_len = self.get_tokenized_length(text)
            if tok_len <= max_length:
                return [text], [[0, len(text)]]
            else:
                if title and title_handling != "none":  # make space for the title in each split text.
                    ltitle = self.get_tokenized_length(title)
                    max_length -= ltitle
                    ind = text.find(title)
                    if ind == 0:
                        text = text[ind + len(title):]

                if self.aligned_on_sentences:
                    if not self.nlp:
                        self.nlp = pyizumo.load(language_code, parsers=['token', 'sentence'])
                    parsed_text = self.nlp(text)

                    tsizes = []
                    begins = []
                    ends = []
                    for sent in parsed_text.sentences:
                        stext = sent.text
                        slen = self.get_tokenized_length(stext)
                        if slen > max_length:
                            too_long = [[t for t in sent.tokens]]
                            too_long[0].reverse()
                            while len(too_long) > 0:
                                tl = too_long.pop(-1)
                                ll = self.get_tokenized_length(text[tl[-1].begin:tl[0].end])
                                if ll <= max_length:
                                    tsizes.append(ll)
                                    begins.append(tl[-1].begin)
                                    ends.append(tl[0].end)
                                else:
                                    if len(tl) > 1:  # Ignore really long words
                                        mid = int(len(tl) / 2)
                                        too_long.extend([tl[:mid], tl[mid:]])
                                    else:
                                        pass
                        else:
                            tsizes.append(slen)
                            begins.append(sent.begin)
                            ends.append(sent.end)

                    intervals = TextTiler.compute_intervals(segment_lengths=tsizes, max_length=max_length, stride=stride)

                    positions = [[begins[p[0]], ends[p[1]]] for p in intervals]
                    texts = [text[p[0]:p[1]] for p in positions]
                else:
                    res = self.tokenizer(max_length=max_length, stride=stride,
                                         return_overflowing_tokens=True, truncation=True)
                    texts = []
                    positions = []
                    end = re.compile(f' {re.escape(tokenizer.sep_token)}$')
                    init_pos=0
                    for split_passage in res['input_ids']:
                        tt = end.sub(
                            "",
                            tokenizer.decode(split_passage).replace(f"{tokenizer.cls_token} ", "")
                        )
                        texts.append(tt)
                        positions.append([init_pos, init_pos+len(tt)])
                        init_pos += stride
                return texts, positions

    MAX_TRIED = 10000
    @staticmethod
    def compute_intervals(segment_lengths: List[int], max_length: int, stride: int) -> List[List[int | Any]]:
        """
        Compute Intervals Method
        This method computes intervals from a list of segment lengths based on a maximum length and a stride. It returns a list of interval ranges.

        Parameters:
        - segment_lengths (List[int]): A list of segment lengths.
        - max_length (int): The maximum length for each interval.
        - stride (int): The stride value for overlapping intervals.

        Returns:
        - List[List[int | Any]]: A list of interval ranges.

        """
        def check_for_cycling(segments_: List[List[int | Any]], prev_start_index_: int) -> None:
            if len(segments_) > 0 and segments_[-1][0] == prev_start_index_:
                raise RuntimeError(f"You have a problem with the splitting - it's cycling!: {segments_[-3:]}")

        def check_excessive_tries(number_of_tries_: int) -> bool:
            if number_of_tries_ > TextTiler.MAX_TRIED:
                print(f"Too many tried - probably something is wrong with the document.")
                return True
            else:
                return False

        current_index = 1
        current_total_length = segment_lengths[0]
        prev_start_index = 0
        segments = []
        number_of_tries = 0

        while current_index < len(segment_lengths):
            if current_total_length + segment_lengths[current_index] > max_length:
                check_for_cycling(segments, prev_start_index)
                if check_excessive_tries(number_of_tries):
                    return segments
                if segments is None:
                    break
                number_of_tries += 1
                segments.append([prev_start_index, current_index - 1])

                if current_index > 1 and segment_lengths[current_index - 1] + segment_lengths[current_index] <= max_length:
                    overlap_index = current_index - 1
                    overlap = 0
                    max_length_tmp = max_length - segment_lengths[current_index]
                    while overlap_index > 0:
                        overlap += segment_lengths[overlap_index]
                        if overlap < stride and overlap + segment_lengths[overlap_index - 1] <= max_length_tmp:
                            overlap_index -= 1
                        else:
                            break
                    current_index = overlap_index
                prev_start_index = current_index
                current_total_length = 0
            else:
                current_total_length += segment_lengths[current_index]
                current_index += 1
        segments.append([prev_start_index, len(segment_lengths) - 1])

        return segments

    def print_product_histogram(self):
        """
        Prints the product ID histogram.

        This method iterates through the product_counts dictionary and prints the product ID along with its count in descending order of the count.

        :param self: The current instance of the class.
        """
        print(f"Product ID histogram:")
        for k in sorted(self.product_counts.keys(), key=lambda x: self.product_counts[x], reverse=True):
            print(f" {k}\t{self.product_counts[k]}")
