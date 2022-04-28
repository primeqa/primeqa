from oneqa.ir.dense.colbert_top.colbert.modeling.tokenization.query_tokenization_xlmr import QueryTokenizerXLMR

def test_tokenizer():
    expected = ([[     0,   9748,   2367,     83,     70,  35166,     32,      2,      1,
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
         0, 0]])



    attend_to_mask_tokens = False
    maxlen = 50
    tokenizer = QueryTokenizerXLMR(maxlen, 'xlm-roberta-base')
    tensorized = tokenizer.tensorize(['what is the answer?', 'is that not completely ridiculously false?'])
    assert(all(tensorized[0][0].numpy() == expected[0][0]))
    #print()

if __name__ == '__main__':
    test_tokenizer()
