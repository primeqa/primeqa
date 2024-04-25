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

    def create_alignment_matrix(self, indices, num_docs, t=5):
        doc_ids = torch.arange(t * num_docs).view(num_docs, t).to(device=indices.device)
        alignment_matrix = torch.zeros_like(indices, dtype=torch.float).to(device=indices.device)

        for j in range(indices.size(1)):
            alignment = torch.isin(indices[:, j], doc_ids[j]).float()
            alignment_matrix[:, j] = alignment
            
        amat_mask, amat_ids = alignment_matrix.max(dim=-1, keepdim=True)
        return amat_mask, amat_ids

    def forward(self, query_ids, doc_ids, query_attention_mask, doc_attention_mask, nway=1, k=55):
        D = self.encoder(doc_ids, attention_mask=doc_attention_mask)[0]
        Q = self.encoder(query_ids, attention_mask=query_attention_mask)[0]

        D = self.bottleneck(D) 
        Q = self.bottleneck(Q) 

        #inner product b/w doc and query token embeddings
        scores = Q.unsqueeze(1) @ D.transpose(1,2).unsqueeze(0) #bxqxdxs
        
        D_mask = doc_attention_mask.repeat(query_ids.size(0), 1, 1)

        scores.transpose(2,3)[~D_mask.bool()] = -99999
    
        ##Qb, Db, Qt = max_scores.shape[:3]
        Qb, Db, Qt, Dt = scores.shape
        
        clubbed_doc_scores = scores.permute(0,2,1,3).flatten(2,3) 
        
        #topk_scores, topk_indices = clubbed_doc_scores.topk(k, -1)
        topk_scores, indices = clubbed_doc_scores.topk(k, -1)

        #remove Query <pad> scores and indices
        #avoid breaking computational graph for backward pass :(
        topk_scores = topk_scores * query_attention_mask.unsqueeze(2) #differentialble

        topk_scores = topk_scores.repeat_interleave(Db, 0).view(Qb, Db, Qt, -1)
        indices = indices.repeat_interleave(Db, 0).view(Qb, Db, Qt, -1)

        #create alignment matrix
        amat_mask, amat_ids = self.create_alignment_matrix(indices, Db, Dt)

        aligned = topk_scores.view(-1, k)[torch.arange(Qb * Db * Qt).to(\
                    device=topk_scores.device), amat_ids.view(-1)].view(amat_ids.shape) * amat_mask
    
        labels = torch.arange(0, Q.size(0), device=Q.device) * nway 
        Z = (aligned > 0.0).float().flatten(2,3).sum(-1).clamp(min=1e-3)
        
        doc_tok_summed_normalized = (1/Z) * aligned.sum(2).squeeze(-1)

        _, predictions = doc_tok_summed_normalized.max(-1)
        accuracy = 100. * (predictions.view(-1) == labels.view(-1)).long().sum()/predictions.size(0)
            
        loss = torch.nn.CrossEntropyLoss()(doc_tok_summed_normalized, labels)

        return loss, accuracy
