import matplotlib.pyplot as plt
from torch_geometric.data import Batch
from datasets.cow_dataset import CoWDataset
import pyvista


def test_dataset_loading(dataset_path, cfg, dataset_name="cow"):
    print(f"Testing dataset loading for {dataset_name}...")
    print("=" * 50)

    train_dataset = CoWDataset(dataset_name=dataset_name, split='train',
                               data_path=dataset_path, config=cfg)
    test_dataset = CoWDataset(dataset_name=dataset_name, split='test',
                              data_path=dataset_path, config=cfg)
    print("✅ Dataset loaded successfully")

    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)} graphs")
    print(f"Test: {len(test_dataset)} graphs")

    print("\nChecking sample data format...")
    sample_data = train_dataset[0]
    print(f"Sample data keys: {sample_data.keys}")
    print(f"Number of nodes: {sample_data.num_nodes}")
    print(f"Number of edges: {sample_data.num_edges}")
    print(f"Node features shape: {sample_data.x.shape}")
    print(f"Edge index shape: {sample_data.edge_index.shape}")
    print(f"Edge attributes shape: {sample_data.edge_attr.shape}")
    print(f"Positions shape: {sample_data.pos.shape}")

    stats = train_dataset.statistics
    print("\nDataset statistics:")
    print(f"Node count distribution: {stats.num_nodes.most_common(5)}")
    print(f"Edge type distribution: {stats.bond_types}")
    print(f"Atom type distribution: {stats.atom_types}")

    visualize_statistics(train_dataset)

    batch = Batch.from_data_list([train_dataset[i] for i in range(3)])
    print(f"Batch node features shape: {batch.x.shape}")
    print(f"Batch edge index shape: {batch.edge_index.shape}")
    print(f"Batch edge attributes shape: {batch.edge_attr.shape}")
    print(f"Batch positions shape: {batch.pos.shape}")
    print("✅ Batch processing works correctly")

    print("\nDataset loading test completed!")


def visualize_statistics(dataset):
    stats = dataset.statistics

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    node_counts = sorted(stats.num_nodes.items(), key=lambda x: x[0])
    x = [k for k, v in node_counts]
    y = [v for k, v in node_counts]
    plt.bar(x, y)
    plt.title("Node Count Distribution")
    plt.xlabel("Number of nodes")
    plt.ylabel("Count")

    plt.subplot(2, 2, 2)
    edge_types = stats.bond_types.numpy()
    plt.bar(range(len(edge_types)), edge_types)
    plt.title("Edge Type Distribution")
    plt.xlabel("Edge type")
    plt.ylabel("Proportion")

    plt.subplot(2, 2, 3)
    bond_lengths = sorted(stats.bond_lengths.items(), key=lambda x: x[0])
    x = [k for k, v in bond_lengths]
    y = [v for k, v in bond_lengths]
    plt.plot(x, y, marker='o')
    plt.title("Bond Length Distribution")
    plt.xlabel("Bond length")
    plt.ylabel("Count")

    plt.subplot(2, 2, 4)
    bond_angles = sorted(stats.bond_angles.items(), key=lambda x: x[0])
    x = [k for k, v in bond_angles]
    y = [v for k, v in bond_angles]
    plt.plot(x, y, marker='o')
    plt.title("Bond Angle Distribution")
    plt.xlabel("Bond angle (radians)")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def visualize_sample_graph(dataset, index=0):
    data = dataset[index]
    plotter = pyvista.Plotter()
    points = data.pos.numpy()
    plotter.add_mesh(pyvista.PolyData(points), color='red', point_size=10)

    edges = data.edge_index.T.numpy()
    for edge in edges:
        line = pyvista.Line(points[edge[0]], points[edge[1]])
        plotter.add_mesh(line, color='blue', line_width=2)

        plotter.add_title(f"Sample Graph (Nodes: {data.num_nodes}, Edges: {data.num_edges})")
        plotter.show()


if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.dataset = 'imagecas'
        args.datapath = r'./data'

        train_dataset = CoWDataset(dataset_name="imagecas", split='train', data_path=args.datapath, config=args)
        visualize_sample_graph(train_dataset)