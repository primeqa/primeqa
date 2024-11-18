import os
import tqdm
import torch

from primeqa.ir.dense.xtr_top.xtr.indexing.loaders import load_doclens


def _load_jsonl(collection_path):
    print("#> Loading collection...")

    collection = []

    with open(collection_path) as f:
        for line_idx, row in tqdm(enumerate(f), desc="Reading Collection"):
            row = json.loads(row)
            passage = row["text"]
            if row["title"] is not None:
                passage = f"{row['title']} {passage}"
            collection.append(passage)
    return collection

def batch(group, bsize, provide_offset=False):
    offset = 0
    while offset < len(group):
        L = group[offset: offset + bsize]
        yield ((offset, L) if provide_offset else L)
        offset += len(L)
    return

def flatten(L):
    # return [x for y in L for x in y]

    result = []
    for _list in L:
        result += _list

    return result

def optimize_ivf(orig_ivf, orig_ivf_lengths, index_path, num_clusters=None, clusters=None):
    print("#> Optimizing IVF to store map from centroids to list of pids..")

    print("#> Building the emb2pid mapping..")
    all_doclens = load_doclens(index_path, flatten=False)

    all_doclens = flatten(all_doclens)
    total_num_embeddings = sum(all_doclens)
    assert total_num_embeddings == orig_ivf.view(-1).size(0)

    emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

    offset_doclens = 0
    for pid, dlength in enumerate(all_doclens):
        emb2pid[offset_doclens: offset_doclens + dlength] = pid
        offset_doclens += dlength

    print("len(emb2pid) =", len(emb2pid))

    #emb2pid = emb2pid[orig_ivf] #this is basically sorted emb2pid as per cluster assignment

    ivf_lengths = [0] * num_clusters
    eids_per_centroid = [torch.tensor([0]) for _ in range(num_clusters)]

    offset = 0
    for cluster, length in tqdm.tqdm(zip(clusters.tolist(), orig_ivf_lengths.tolist())):
        eids = orig_ivf[offset: offset+length]
        #eids_per_centroid.append(eids)
        eids_per_centroid[cluster] = eids
        #ivf_lengths.append(eids.size(0))
        ivf_lengths[cluster] = eids.size(0)
        offset += length

    ivf = torch.cat(eids_per_centroid)
    ivf_lengths = torch.tensor(ivf_lengths)

    ivf_path = os.path.join(index_path, 'ivf.eid.pt')
    torch.save((ivf, ivf_lengths), ivf_path)

    emb2pid_path = os.path.join(index_path, 'emb2pid.pt')
    torch.save(emb2pid, emb2pid_path)

    print(f"#> Saved IVF to {ivf_path}")
    print(f"#> Saved EMB2PID to {emb2pid_path}")

    return ivf, ivf_lengths
