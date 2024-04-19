import numpy as np

import torch

from primeqa.ir.dense.xtr_top.xtr.search.strided_tensor import StridedTensor
from primeqa.ir.dense.xtr_top.xtr.search.utils.xtr_inference import xtr_inference
from primeqa.ir.dense.xtr_top.xtr.search.strided_tensor_core import _create_mask, _create_view


class CandidateGeneration:

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    def get_cells(self, Q, ncells):
        scores = (self.codec.centroids @ Q.T).transpose(0,1)
        #NOTE handling empty clusters
        if self.empty_clusters.size(0):
            if self.use_gpu:
                scores[:, self.empty_clusters] = torch.tensor(-999999).half()
            else:
                scores[:, self.empty_clusters] = torch.tensor(-999999).float()
        if ncells == 1:
            cells = scores.argmax(dim=1, keepdim=True)
        else:
            cells = scores.topk(ncells, dim=1, sorted=False)

        return cells.indices, cells.values

    def create_alignment_matrix(self, indices, doc_ids):
        alignment_matrix = torch.zeros_like(indices, dtype=torch.float).to(device=indices.device)
        for j in range(doc_ids.size(-1)):
            alignment_matrix[:, j] = torch.isin(indices[:, j], doc_ids[j]).float()
    
        amat_mask, amat_ids = alignment_matrix.max(dim=-1, keepdim=True)
        return amat_mask, amat_ids
 
    def generate_candidate_eids(self, cells):
        max_num_embs = float('-inf')
        all_cell_embs_range = [] #self.ivf_ranges[cell].cuda() for cell in cells]
        for cell in cells:
            cell_embs_range = torch.arange(self.ivf_lengths_prev[cell], self.ivf_lengths[cell])
            nembs = cell_embs_range.size(0)
            if nembs > max_num_embs:
                max_num_embs = nembs
            if self.use_gpu:
                cell_embs_range = cell_embs_range.cuda()
            all_cell_embs_range.append(cell_embs_range)
           
        eids = torch.index_select(self.ivf, 0, torch.cat(all_cell_embs_range))
        return eids, max_num_embs

    def generate_candidate_scores(self, Q, eids):
        E = self.lookup_eids(eids, out_device=self.use_gpu)
        if self.use_gpu:
            E = E.cuda()
        score = Q @ E.T
        return score

    def generate_candidates(self, config, Q, mask=None, k=10):
        ncells = config.ncells

        Q = Q.reshape(-1, Q.size(-1))
        Q = Q[mask.reshape(-1).bool()]
        cells, centroid_scores = self.get_cells(Q, ncells)

        count = 0
        batch_predictions = []
        for imask in mask:
            num_tokens = imask.sum().item()
            icells = cells[count:count+num_tokens].flatten().unique(sorted=False)
            eids, _ = self.generate_candidate_eids(icells)
            pids = self.emb2pid[eids]
            scores = self.generate_candidate_scores(Q[count:count+num_tokens], eids)
            k_scores, k_indices = scores.topk(k=min(config.ndocs, scores.size(-1)))
            k_indices = pids[k_indices.flatten()].view(k_indices.shape).cpu().numpy().astype(np.int64)
            k_indices = self.mappings['doc_indices'][k_indices.flatten()].reshape(k_indices.shape).astype(np.int64)
            pred_doc_ids = xtr_inference(k_scores.cpu().numpy().astype(np.float64), k_indices)
            pred_doc_ids = {self.mappings['doc_strings'][d]:s for d, s in sorted(pred_doc_ids.items(), key=lambda x:x[1], reverse=True)[:k]}
            batch_predictions.append(pred_doc_ids)
            count += num_tokens
        return batch_predictions
