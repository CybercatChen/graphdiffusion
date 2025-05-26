import os
import pickle
from collections import Counter

from matplotlib import pyplot as plt


def save_pickle(array, path):
    with open(path, 'wb') as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_counter_as_hist(data_counter, second_counter, x_label=None, y_label='Probability', name=None):
    data_dict = dict(data_counter)
    second_dict = dict(second_counter)
    homogenize_keys(data_dict=data_dict, second_dict=second_dict, fill_value=0)
    new_data_dict = sort_dict_by_keys(data_dict)
    new_second_dict = sort_dict_by_keys(second_dict)
    index_locations = list(new_second_dict.keys())
    plt.bar(index_locations, new_data_dict.values(), align='center', alpha=0.5, color='blue', edgecolor='none')
    plt.bar(index_locations, new_second_dict.values(), align='center', alpha=0.5, color='orange', edgecolor='none')
    plt.xticks(index_locations, index_locations)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if name is not None:
        plt.title(name)
        plt.savefig(os.path.join(os.getcwd(), f"{name}.png"))
    plt.show()
    plt.clf()


def plot_list_as_hist(data, x_label=None, y_label='Probability', second_list=None, name=None):
    index_locations = list(range(len(data)))
    plt.bar(index_locations, data, align='center', alpha=0.5, color='blue', edgecolor='none', label='gt')
    if second_list is not None:
        plt.bar(index_locations, second_list, align='center', alpha=0.5, color='orange', edgecolor='none',
                label='synth')
    plt.xticks(index_locations, index_locations)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if name is not None:
        plt.legend()
        plt.title(name)
        plt.savefig(os.path.join(os.getcwd(), f"{name}.png"))
    plt.show()
    plt.clf()


def homogenize_keys(data_dict, second_dict, fill_value=None):
    all_keys = set(data_dict.keys()).union(second_dict.keys())
    for key in all_keys:
        if key not in data_dict.keys():
            data_dict[key] = [] if fill_value is None else fill_value
        if key not in second_dict.keys():
            second_dict[key] = [] if fill_value is None else fill_value


def sort_dict_by_keys(original_dict):
    keys = list(original_dict.keys())
    keys.sort()
    sorted_dict = {i: original_dict[i] for i in keys}
    return sorted_dict


def homogenize_dicts(data_dict, second_dict):
    homogenize_keys(data_dict, second_dict)
    new_data_dict = sort_dict_by_keys(data_dict)
    new_second_dict = sort_dict_by_keys(second_dict)
    for (k1, l1), (k2, l2) in zip(new_data_dict.items(), new_second_dict.items()):
        assert k1 == k2, "Keys in random order"
        if isinstance(l1, Counter):
            l1 = list(l1.values())
        if isinstance(l2, Counter):
            l2 = list(l2.values())
        max_len = max(len(l1), len(l2))
        if len(l1) < max_len:
            l1 = l1 + (max_len - len(l1)) * [0]
            new_data_dict[k1] = l1
        if len(l2) < max_len:
            l2 = l2 + (max_len - len(l2)) * [0]
            new_second_dict[k2] = l2
    return new_data_dict, new_second_dict


def plot_list_of_dict_as_hist(data_dict, second_dict=None, name=None):
    num_keys = len(data_dict)
    num_cols = 2  # Number of columns per row
    num_rows = (num_keys + num_cols - 1) // num_cols  # Ceiling division

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
    fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing
    if second_dict is not None:
        data_dict, second_dict = homogenize_dicts(data_dict, second_dict)
    for i, (key, value_list) in enumerate(data_dict.items()):
        row_idx = i // num_cols
        col_idx = i % num_cols

        index_locations = list(range(len(value_list)))
        axs[row_idx, col_idx].bar(index_locations, value_list, align='center', alpha=0.5, color='blue',
                                  edgecolor='none', label='gt')
        if second_dict is not None:
            axs[row_idx, col_idx].bar(index_locations, second_dict[key], align='center', alpha=0.5, color='orange',
                                      edgecolor='none', label='generated')
        axs[row_idx, col_idx].set_xlabel('Values')
        axs[row_idx, col_idx].set_ylabel('Frequency')
        axs[row_idx, col_idx].set_title(f'Histogram Plot ({key})')

    for i in range(num_keys, num_rows * num_cols):
        row_idx = i // num_cols
        col_idx = i % num_cols
        fig.delaxes(axs[row_idx, col_idx])

    plt.tight_layout()
    if name is not None:
        plt.legend()
        plt.title(name)
        plt.savefig(os.path.join(os.getcwd(), f"{name}.png"))
    # Show the plots
    plt.show()
    plt.clf()


class Statistics:
    def __init__(self, num_nodes, atom_types, bond_types, bond_lengths, bond_angles,
                 charge_types, betti_vals=None, cycles_cond_N=None, cc_cond_N=None, deg_cond_N=None):
        self.num_nodes = num_nodes
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.bond_lengths = bond_lengths
        self.bond_angles = bond_angles
        self.betti_vals = betti_vals
        self.charge_types = charge_types
        self.cycles_cond_N = cycles_cond_N
        self.cc_cond_N = cc_cond_N
        self.deg_cond_N = deg_cond_N

    def __repr__(self):
        return f"atom types: {self.atom_types}\nbond types: {self.bond_types}\nbond lengths: {self.bond_lengths}\n" \
               f"bond angles: {self.bond_angles}\n betti vals: {self.betti_vals}\n charge types {self.charge_types}\n" \
               f"connected component condn: {self.cc_cond_N}\n degree condition:{self.deg_cond_N}"
