import math
from collections.abc import Mapping, Sequence
from typing import Union, List, Optional

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData

try:
    from torch_geometric.data import LightningDataset
except ImportError:
    from torch_geometric.data.lightning import LightningDataset


def effective_batch_size(max_size, reference_batch_size, reference_size=20, sampling=False):
    x = reference_batch_size * (reference_size / max_size) ** 2
    return math.floor(1.8 * x) if sampling else math.floor(x)


# class AdaptiveCollater:
#     def __init__(self, follow_batch, exclude_keys, reference_batch_size):
#         self.follow_batch = follow_batch
#         self.exclude_keys = exclude_keys
#         self.reference_bs = reference_batch_size
#
#     def __call__(self, batch):
#         elem = batch[0]
#         if isinstance(elem, BaseData):
#             to_keep = []
#             graph_sizes = []
#
#             for e in batch:
#                 e: BaseData
#                 graph_sizes.append(e.num_nodes)
#
#             m = len(graph_sizes)
#             graph_sizes = torch.Tensor(graph_sizes)
#             srted, argsort = torch.sort(graph_sizes)
#             random = torch.randint(0, m, size=(1, 1)).item()
#             max_size = min(srted.max().item(), srted[random].item() + 5)
#             max_size = max(max_size, 9)  # The batch sizes may be huge if the graphs happen to be tiny
#
#             ebs = effective_batch_size(max_size, self.reference_bs)
#
#             max_index = torch.nonzero(srted <= max_size).max().item()
#             min_index = max(0, max_index - ebs)
#             indices_to_keep = set(argsort[min_index: max_index + 1].tolist())
#             if max_index < ebs:
#                 for index in range(max_index + 1, m):
#                     # Check if we could add the graph to the list
#                     size = srted[index].item()
#                     potential_ebs = effective_batch_size(size, self.reference_bs)
#                     if len(indices_to_keep) < potential_ebs:
#                         indices_to_keep.add(argsort[index].item())
#
#             for i, e in enumerate(batch):
#                 e: BaseData
#                 if i in indices_to_keep:
#                     to_keep.append(e)
#
#             new_batch = Batch.from_data_list(to_keep, self.follow_batch, self.exclude_keys)
#             return new_batch
#
#         elif True:
#             raise NotImplementedError("Only supporting BaseData for now")
#         elif isinstance(elem, torch.Tensor):
#             return default_collate(batch)
#         elif isinstance(elem, float):
#             return torch.tensor(batch, dtype=torch.float)
#         elif isinstance(elem, int):
#             return torch.tensor(batch)
#         elif isinstance(elem, str):
#             return batch
#         elif isinstance(elem, Mapping):
#             return {key: self([data[key] for data in batch]) for key in elem}
#         elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
#             return type(elem)(*(self(s) for s in zip(*batch)))
#         elif isinstance(elem, Sequence) and not isinstance(elem, str):
#             return [self(s) for s in zip(*batch)]
#
#     def collate(self, batch):
#         return self(batch)
#
#
# class AdaptiveDataLoader(torch.utils.data.DataLoader):
#     def __init__(
#             self,
#             dataset: Union[Dataset, List[BaseData]],
#             batch_size: int = 1,
#             reference_batch_size: int = 1,
#             shuffle: bool = False,
#             follow_batch: Optional[List[str]] = None,
#             exclude_keys: Optional[List[str]] = None,
#             **kwargs,
#     ):
#         if 'collate_fn' in kwargs:
#             del kwargs['collate_fn']
#
#         self.follow_batch = follow_batch
#         self.exclude_keys = exclude_keys
#
#         super().__init__(
#             dataset,
#             batch_size,
#             shuffle,
#             collate_fn=AdaptiveCollater(follow_batch, exclude_keys, reference_batch_size=reference_batch_size),
#             **kwargs,
#         )
#
#
# class AdaptiveLightningDataset(LightningDataset):
#     def __init__(
#             self,
#             train_dataset: Dataset,
#             val_dataset: Optional[Dataset] = None,
#             test_dataset: Optional[Dataset] = None,
#             batch_size: int = 1,
#             reference_batch_size: int = 1,
#             num_workers: int = 0,
#             **kwargs,
#     ):
#         self.reference_batch_size = reference_batch_size
#         super().__init__(
#             train_dataset=train_dataset,
#             val_dataset=val_dataset,
#             test_dataset=test_dataset,
#             batch_size=batch_size,
#             num_workers=num_workers,
#             **kwargs,
#         )
#
#         self.train_dataset = train_dataset
#         self.val_dataset = val_dataset
#         self.test_dataset = test_dataset
#
#     def dataloader(self, dataset: Dataset, shuffle: bool = False, **kwargs) -> AdaptiveDataLoader:
#         return AdaptiveDataLoader(dataset, reference_batch_size=self.reference_batch_size,
#                                   shuffle=shuffle, **self.kwargs)
