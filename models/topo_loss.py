import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
import torch
import networkx as nx
import numpy as np


def wrcf_from_adj(adj: torch.Tensor) -> gd.SimplexTree:
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Input adjacency matrix must be square.")
    n = adj.shape[0]
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            weight = adj[i, j].item()
            if weight > 0:
                G.add_edge(i, j, weight=weight)
    st = gd.SimplexTree()
    for v in G.nodes():
        st.insert([v], filtration=0)
    distinct_weights = np.unique([i[2] for i in G.edges.data("weight")])[::-1]
    for w in distinct_weights:
        subg = G.edge_subgraph([(u, v) for u, v, _w in G.edges.data("weight") if _w >= w])
        for clique in nx.find_cliques(subg):
            st.insert(clique, filtration=1 / w)
    return st


def get_persistence_diagram(adj: torch.Tensor, dim: int, card: int) -> torch.Tensor:
    st = wrcf_from_adj(adj)
    pairs = st.persistence_pairs()
    indices, pers = [], []
    for s1, s2 in pairs:
        if len(s1) == dim + 1 and len(s2) > 0:
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [s1[v] for v in np.unravel_index(np.argmax(adj[l1, :][:, l1]), [len(s1), len(s1)])]
            i2 = [s2[v] for v in np.unravel_index(np.argmax(adj[l2, :][:, l2]), [len(s2), len(s2)])]
            indices += i1
            indices += i2
            pers.append(st.filtration(s2) - st.filtration(s1))
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1, 4])[perm][::-1, :].flatten())
    indices = indices[:4 * card] + [0 for _ in range(0, max(0, 4 * card - len(indices)))]
    ids = torch.tensor(indices, dtype=torch.int32)
    if dim > 0:
        indices = ids.view([2 * card, 2]).long()
        dgm = adj[indices[:, 0], indices[:, 1]].view(card, 2)
    else:
        indices = ids.view([2 * card, 2])[1::2, :].long()
        dgm = torch.cat([torch.zeros(card, 1), adj[indices[:, 0], indices[:, 1]].view(card, 1)], dim=1)
    return dgm


def compute_topo_loss(pred_adj: torch.Tensor, gt_adj: torch.Tensor, card: int = 50, lam: float = 1.0) -> torch.Tensor:
    dgm_pred_0 = get_persistence_diagram(pred_adj, dim=0, card=card)
    dgm_pred_1 = get_persistence_diagram(pred_adj, dim=1, card=card)
    dgm_gt_0 = get_persistence_diagram(gt_adj, dim=0, card=card)
    dgm_gt_1 = get_persistence_diagram(gt_adj, dim=1, card=card)
    loss0 = wasserstein_distance(dgm_pred_0, dgm_gt_0, order=1, enable_autodiff=True, keep_essential_parts=False)
    loss1 = wasserstein_distance(dgm_pred_1, dgm_gt_1, order=1, enable_autodiff=True, keep_essential_parts=False)
    return lam * (loss0 + loss1)

