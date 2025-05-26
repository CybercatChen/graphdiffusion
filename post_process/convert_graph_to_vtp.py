import os
import shutil

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, contains_isolated_nodes, remove_isolated_nodes
from tqdm import tqdm

from datasets.dataset_utils import load_pickle, save_pickle
from post_process.post_process_utils import save_as_vtp


def apply_padding_offset(translated_nodes, new_min, new_max, selected_edges, selected_edge_attr):
    new_com, orig_com = 0.5, 0
    offset = new_com - orig_com
    translated_nodes += offset
    translated_nodes = torch.clamp(translated_nodes, min=new_min, max=new_max)
    return translated_nodes, selected_edges, selected_edge_attr


def align_graph_nodes_save_vtp(load_dir, discrete_graph_file, cont_graph_file,
                               seg_has_pads, patch_min, patch_max):
    pyG_list = load_pickle(path=os.path.join(load_dir, discrete_graph_file))
    new_data_dict = {}
    for idx, pyG_dataset in tqdm(enumerate(pyG_list), total=len(pyG_list)):
        nodes, edges, edge_attr = pyG_dataset.x, pyG_dataset.edge_index, pyG_dataset.edge_attr
        edges, edge_attr = remove_self_loops(edge_index=edges, edge_attr=edge_attr)

        if len(edge_attr) == 0:
            print(f"Generated 0 attribute graph {pyG_dataset=}")
            continue
        if contains_isolated_nodes(edge_index=edges, num_nodes=nodes.size(0)):
            print("Found graph with isolated nodes")
            edges, edge_attr, mask = remove_isolated_nodes(edge_index=edges, edge_attr=edge_attr,
                                                           num_nodes=nodes.size(0))
            nodes = nodes[mask]
            print(f"Num node decrease by {(mask == False).sum()}")
        if len(edge_attr) == 0:
            print(f"Generated 0 attribute graph {pyG_dataset=}".Skipping)
            continue
        selected_edges, selected_edge_attr = edges, edge_attr
        if torch.any(selected_edge_attr <= 0):
            print(f"{selected_edge_attr=}")
        if seg_has_pads:
            nodes, selected_edges, selected_edge_attr = apply_padding_offset(nodes, patch_min,
                                                                             patch_max, selected_edges,
                                                                             selected_edge_attr)
            if len(selected_edge_attr) == 0:
                print("Translation caused zero length edge")
                print(f"Skipping {selected_edges=} and {selected_edge_attr=}")
                continue

        save_as_vtp(base_dir=load_dir, nodes=nodes, edge_index=selected_edges,
                    edge_attr=selected_edge_attr, idx=f"{idx}")
        new_data_dict[idx] = Data(x=nodes, edge_index=selected_edges, edge_attr=selected_edge_attr)
    save_pickle(array=new_data_dict, path=os.path.join(load_dir, cont_graph_file))


if __name__ == '__main__':
    load_dir = r'/home/siqichen/vessel_dif/logs/march2025-05-26/17-45-43-graph-vessel-model'
    torch_geometric.seed_everything(42)
    dest_folder = os.path.join(load_dir, "synthetic_data", "vtp")
    if os.path.exists(dest_folder):
        print("Cleaning destination folder")
        shutil.rmtree(dest_folder)
    seg_has_pads = "imagecas" not in load_dir
    align_graph_nodes_save_vtp(load_dir=load_dir,
                               discrete_graph_file='generated_samples.pkl',
                               cont_graph_file='processed_samples.pkl',
                               seg_has_pads=seg_has_pads, patch_min=5 / 64,
                               patch_max=1 - 5 / 64)
