import os
import numpy as np
import networkx as nx
from plyfile import PlyData
import pyvista as pv
from glob import glob


def read_ply(filename):
    data = PlyData.read(filename)
    vertex_data = data['vertex']
    vertices = np.array([[vertex[prop.name] for prop in vertex_data.properties] for vertex in vertex_data])
    edges = np.array([[edge[0], edge[1]] for edge in data['edge']])
    return vertices, edges


def ply_to_graph(filename):
    vertices, edges = read_ply(filename)
    G = nx.Graph()
    for i, vertex in enumerate(vertices):
        G.add_node(i, x=vertex[0], y=vertex[1], z=vertex[2], attr=vertex[3])  # Only first 4 features used
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    return G


def find_roots_by_attr(graph):
    roots = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        attr_dict = {}
        for n in subgraph.nodes:
            attr = subgraph.nodes[n]['attr']
            attr_dict.setdefault(attr, []).append(n)

        # 找出有相同属性的两个节点
        for attr, nodes in attr_dict.items():
            if len(nodes) == 2:
                # 选一个度为1的叶子节点作为根
                for n in nodes:
                    if subgraph.degree[n] == 1:
                        roots.append(n)
                        break
    return roots


def assign_edge_attrs(graph, roots):
    for root in roots:
        for parent, child in nx.bfs_edges(graph, source=root):
            graph.edges[parent, child]['attr'] = graph.nodes[child]['attr']


def graph_to_vtp(graph, out_path):
    points = []
    lines = []
    edge_attr = []

    index_map = {node: i for i, node in enumerate(graph.nodes)}
    for node in graph.nodes:
        x, y, z = graph.nodes[node]['x'], graph.nodes[node]['y'], graph.nodes[node]['z']
        points.append([x, y, z])

    for u, v in graph.edges:
        lines.append([2, index_map[u], index_map[v]])
        edge_attr.append(graph.edges[u, v].get('attr', 0.0))

    poly = pv.PolyData()
    poly.points = np.array(points)
    poly.lines = np.array(lines)
    poly['edge_attr'] = np.array(edge_attr)
    out_path = out_path.replace('.ply', '.vtp')
    poly.save(out_path)


def process_ply_folder(folder_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(folder_path):
        if file.endswith(".ply"):
            file_path = os.path.join(folder_path, file)
            graph = ply_to_graph(file_path)

            roots = find_roots_by_attr(graph)
            assign_edge_attrs(graph, roots)
            vtp_name = os.path.splitext(file)[0] + '.vtp'
            vtp_path = os.path.join(output_dir, vtp_name)
            graph_to_vtp(graph, vtp_path)
            print(f"Saved: {vtp_path}")


def discretize_edges(vtp_dir, output_dir, num_bins=7, method='quantile'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vtp_files = glob(os.path.join(vtp_dir, '*.vtp'))

    for file in vtp_files:
        mesh = pv.read(file)
        edge_attr = np.array(mesh['edge_attr'])

        if method == 'quantile':
            bins = np.quantile(edge_attr, np.linspace(0, 1, num_bins + 1))
        elif method == 'uniform':
            bins = np.linspace(np.min(edge_attr), np.max(edge_attr), num_bins + 1)
        else:
            raise ValueError("method must be 'quantile' or 'uniform'")

        bins = np.unique(bins)
        if len(bins) - 1 < num_bins:
            print(f"Note: {file} only has {len(bins) - 1} unique bins.")
        edge_discrete = np.digitize(edge_attr, bins[1:], right=False)

        mesh['edge_attr'] = edge_discrete.astype(np.uint8)

        out_name = os.path.basename(file)
        out_path = os.path.join(output_dir, out_name)
        mesh.save(out_path)
        print(f"Saved discrete VTP: {out_path}")


if __name__ == '__main__':
    # process_ply_folder(r'D:\final_data\Dataset\March2013\key_graph_norm', r'D:\final_data\Dataset\March2013\vtp_data')
    discretize_edges(r'D:\final_data\Dataset\March2013\vtp_data', r'D:\final_data\Dataset\March2013\vtp_data_discrete',
                     num_bins=12, method='uniform')
