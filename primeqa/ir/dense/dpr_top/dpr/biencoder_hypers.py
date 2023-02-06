from primeqa.ir.dense.dpr_top.torch_util.hypers_base import HypersBase
import os


class BiEncoderHypers(HypersBase):
    def __init__(self):
        super().__init__()
        # use 'facebook/dpr-ctx_encoder-multiset-base', 'facebook/dpr-question_encoder-multiset-base'
        # or 'facebook/dpr-question_encoder-single-nq-base', 'facebook/dpr-ctx_encoder-single-nq-base'
        self.qry_encoder_name_or_path = 'facebook/dpr-question_encoder-multiset-base'
        self.ctx_encoder_name_or_path = 'facebook/dpr-ctx_encoder-multiset-base'
        self.encoder_gpu_train_limit = 8  # max number of instances to encode (-1) to disable gradient checkpointing

        # https://github.com/facebookresearch/DPR#2-biencoderretriever-training-in-single-set-mode
        self.max_grad_norm = 2.0  # default from HypersBase is 1.0
        self.learning_rate = 2e-05  # default from HypersBase is 5e-5

        self.seq_len_q = 64   # max length for query
        self.seq_len_c = 128  # max length for context
        self.debug_location = ''  # where to save debug info
        self.sample_negative_from_top_k = 1
        self.force_confict_free_batches = False  # we check that batches do not contain instances s.t. batch negatives will actually be positives
        self.training_data_type = 'dpr'
        self.collection = '' # used with training_data_type == 'num_triples'
        self.queries = ''    # used with training_data_type == 'num_triples'

        self.__required_args__ = []

    def _post_init(self):
        super()._post_init()
