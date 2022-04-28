from oneqa.ir.dense.colbert_top.colbert.modeling.tokenization.query_tokenization import QueryTokenizer
from oneqa.ir.dense.colbert_top.colbert.modeling.tokenization.query_tokenization_xlmr import QueryTokenizerXLMR
from oneqa.ir.dense.colbert_top.colbert.modeling.tokenization.doc_tokenization import DocTokenizer
from oneqa.ir.dense.colbert_top.colbert.modeling.tokenization.doc_tokenization_xlmr import DocTokenizerXLMR

def test_tokenizer():

    expected_all = {
    'query_tokenization' :
        ([[ 101,    1, 2054, 2003, 1996, 3437, 1029,  102,  103,  103,  103,  103,
          103,  103,  103,  103,  103,  103,  103,  103,  103,  103,  103,  103,
          103,  103,  103,  103,  103,  103,  103,  103,  103,  103,  103,  103,
          103,  103,  103,  103,  103,  103,  103,  103,  103,  103,  103,  103,
          103,  103],
        [ 101,    1, 2003, 2008, 2025, 3294, 9951, 2135, 6270, 1029,  102,  103,
          103,  103,  103,  103,  103,  103,  103,  103,  103,  103,  103,  103,
          103,  103,  103,  103,  103,  103,  103,  103,  103,  103,  103,  103,
          103,  103,  103,  103,  103,  103,  103,  103,  103,  103,  103,  103,
          103,  103]],
        [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0]]),
    'query_tokenization_xlmr' :
        ([[     0,   9748,   2367,     83,     70,  35166,     32,      2,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1],
        [     0,   9748,     83,    450,    959,  64557, 236873,    538,  98320,
             32,      2,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1]],
    [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0]]),
    'doc_tokenization' :
        ([[ 101,    2, 2054, 2003, 1996, 3437, 1029,  102,    0,    0,    0],
        [ 101,    2, 2003, 2008, 2025, 3294, 9951, 2135, 6270, 1029,  102]],
        [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
    'doc_tokenization_xlmr' :
      ([[     0,   9749,   2367,     83,     70,  35166,     32,      2,      1,
              1,      1],
        [     0,   9749,     83,    450,    959,  64557, 236873,    538,  98320,
             32,      2]], [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    }

    question = 'what is the answer?'
    answer = 'is that not completely ridiculously false?'

    maxlen = 50
    attend_to_mask_tokens = False
    tokenizer = QueryTokenizer(maxlen, 'bert-base-uncased', attend_to_mask_tokens)
    tensorized = tokenizer.tensorize([question, answer])
    expected = expected_all['query_tokenization']
    for ii in range(len(expected)):
        for jj in range(len(expected[ii])):
            assert(all(tensorized[ii][jj].numpy() == expected[ii][jj]))

    maxlen = 50
    tokenizer = QueryTokenizerXLMR(maxlen, 'xlm-roberta-base')
    tensorized = tokenizer.tensorize([question, answer])
    expected = expected_all['query_tokenization_xlmr']
    for ii in range(len(expected)):
        for jj in range(len(expected[ii])):
            assert(all(tensorized[ii][jj].numpy() == expected[ii][jj]))

    maxlen = 180
    tokenizer = DocTokenizer(maxlen, 'bert-base-uncased')
    tensorized = tokenizer.tensorize([question, answer])
    expected = expected_all['doc_tokenization']
    for ii in range(len(expected)):
        for jj in range(len(expected[ii])):
            assert(all(tensorized[ii][jj].numpy() == expected[ii][jj]))

    maxlen = 50
    tokenizer = DocTokenizerXLMR(maxlen, 'xlm-roberta-base')
    tensorized = tokenizer.tensorize([question, answer])
    expected = expected_all['doc_tokenization_xlmr']
    for ii in range(len(expected)):
        for jj in range(len(expected[ii])):
            assert(all(tensorized[ii][jj].numpy() == expected[ii][jj]))

    print()

if __name__ == '__main__':
    test_tokenizer()
