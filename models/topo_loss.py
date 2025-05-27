import gudhi as gd
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from gudhi.wasserstein import wasserstein_distance


def wrcf(G: nx.Graph) -> gd.SimplexTree:
    st = gd.SimplexTree()
    for v in G.nodes():
        st.insert([v], filtration=0.0)

    for u, v, w in G.edges(data="weight"):
        if w >= 1:
            st.insert([u, v], filtration=1.0)
    return st


def compute_wrcf_intervals(adj: np.ndarray, dim: int) -> torch.Tensor:
    G = nx.from_numpy_array(adj)
    st = wrcf(G)
    st.persistence()
    intervals = np.array(st.persistence_intervals_in_dimension(dim))
    if intervals.size == 0:
        intervals = np.array([[0.0, 0.0]])
    return torch.from_numpy(intervals).float()


def compute_topo_loss(pre_adj, gt_adj, dim):
    pre_pd = compute_wrcf_intervals(pre_adj.cpu().numpy(), dim).cuda(pre_adj.device)
    gt_pd = compute_wrcf_intervals(gt_adj.cpu().numpy(), dim).cuda(gt_adj.device)

    loss = wasserstein_distance(
        pre_pd, gt_pd,
        order=1,
        enable_autodiff=True,
        keep_essential_parts=False
    )
    return loss
