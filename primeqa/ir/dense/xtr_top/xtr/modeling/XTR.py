import copy

import torch
from torch import nn

from transformers import T5EncoderModel


class XTR(T5EncoderModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        #create bottleneck for indexing
        self.bottleneck = torch.nn.Linear(config.d_model, config.dim, bias=False) 

        self.init_weights()

    def encode(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        return self.bottleneck(outputs[0])

    def forward(self, query_ids, doc_ids, query_attention_mask, doc_attention_mask, nway=1, k=55):
        D = self.encoder(doc_ids, attention_mask=doc_attention_mask)[0]
        Q = self.encoder(query_ids, attention_mask=query_attention_mask)[0]

        D = self.bottleneck(D) 
        Q = self.bottleneck(Q) 

        #inner product b/w doc and query token embeddings
        scores = Q.unsqueeze(1) @ D.transpose(1,2).unsqueeze(0) #bxqxdxs
        
        D_mask = doc_attention_mask.repeat(query_ids.size(0), 1, 1)

        #replace Doc <pad> scores with a large -ve number
        scores.transpose(2,3)[~D_mask.bool()] = -99999
    
        ##Qb, Db, Qt = max_scores.shape[:3]
        Qb, Db, Qt, Dt = scores.shape
        
        clubbed_doc_scores = scores.permute(0,2,1,3).flatten(2,3) 
        
        topk_scores, topk_indices = clubbed_doc_scores.topk(k, -1)

        #create a boolen vector of True for all positions
        alignment_mask = torch.ones_like(clubbed_doc_scores, dtype=torch.bool)

        #mask Query <pad> scores and indices
        topk_scores = topk_scores * query_attention_mask.unsqueeze(2) 

        #mask the topk positions to 0
        alignment_mask.scatter_(-1, topk_indices, 0)
        
        #change to 0 all the non-topk position scores, leaving topk scores intact
        clubbed_doc_scores.masked_fill(alignment_mask, 0)
        
        #change the clubbed scores to original shape of QbxQtxDbxDt
        topk_scores_max = clubbed_doc_scores.view(Qb,Qt,Db,-1).max(-1).values
    
        #get the normalizer for each doc score as the number of non-zeros scores per doc
        #clamp 0's with some small number to avoid division by zero errors
        Z = (topk_scores_max > 0.0).float().sum(1).clamp(min=1e-3)
        
        #normalize scores
        doc_tok_summed_normalized = (1/Z) * topk_scores_max.sum(1)
                    
        #create labels
        labels = torch.arange(0, Q.size(0), device=Q.device) * nway 

        #compute training accuracy
        _, predictions = doc_tok_summed_normalized.max(-1)
        accuracy = 100. * (predictions.view(-1) == labels.view(-1)).long().sum()/predictions.size(0)

        #compute loss
        loss = torch.nn.CrossEntropyLoss()(doc_tok_summed_normalized, labels)

        return loss, accuracy
