import unicodedata
import regex


class HasAnswerChecker(object):

    ''' 
        in http://www.regular-expressions.info/unicode.html#prop
            \p{L} or \p{Letter}: any kind of letter from any language.
            \p{M} or \p{Mark}: a character intended to be combined with another character (e.g. accents, umlauts, enclosing boxes, etc.).
            \p{N} or \p{Number}: any kind of numeric character in any script
            \p{Z} or \p{Separator}: any kind of whitespace or invisible separator
            \p{C} or \p{Other}: invisible control characters and unused code points. 
    '''
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS_SEP = r'[^\p{Z}\p{C}]'
    TOKENIZER_REGEXP = regex.compile(
            '(%s)|(%s)' % (ALPHA_NUM, NON_WS_SEP),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE)

    # canonical decomposition, and translates each character into its decomposed form
    def normalize(self, text):
        return unicodedata.normalize('NFD', text)

    # Simple whitespace tokenization
    def tokenize(self, text):
        tokens = [m.group().lower() for m in self.TOKENIZER_REGEXP.finditer(text)]
        return tokens

    # return true if any of the answer strings is found in the text
    def has_answer(self, answers, text):
        normalized_text = self.normalize(text)
        tokenized_text = self.tokenize(normalized_text)

        for answer in answers:
            normalized_answer = self.normalize(answer)
            tokenized_answer = self.tokenize(normalized_answer)
            
            for i in range(0, len(tokenized_text) + 1):
                if tokenized_answer == tokenized_text[i: i + len(tokenized_answer)]:
                    return True

        return False


