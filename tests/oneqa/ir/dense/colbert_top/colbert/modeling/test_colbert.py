from oneqa.ir.dense.colbert_top.colbert.modeling.hf_colbert import HF_ColBERT
from oneqa.ir.dense.colbert_top.colbert.modeling.hf_colbert_xlmr import HF_ColBERT_XLMR
from oneqa.ir.dense.colbert_top.colbert.modeling.colbert import ColBERT

def test_colbert():
    colbert = ColBERT(name='bert-base-uncased', colbert_config=None)

    colbert = ColBERT(name='xlm-roberta-base', colbert_config=None)

    print()

if __name__ == '__main__':
    test_colbert()
