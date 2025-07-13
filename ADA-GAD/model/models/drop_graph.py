from typing import Optional, Tuple

import torch

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None
from torch import Tensor

from torch_geometric.deprecation import deprecated
from torch_geometric.typing import OptTensor
from torch_geometric.utils import degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils import subgraph

def dropout_subgraph(edge_index: Tensor, p: float = 0.2, walks_per_node: int = 1,
                 walk_length: int = 3, num_nodes: Optional[int] = None,
                 is_sorted: bool = False,
                 training: bool = True,return_subgraph:bool=True) -> Tuple[Tensor, Tensor, Tensor]:
    """稀疏实现：仅返回采样后的edge_index和edge_mask，subgraph_mask直接返回采样到的节点对的edge_index稀疏列表，不再生成稠密矩阵。"""
    if p < 0. or p > 1.:
        raise ValueError(f'Sample probability has to be between 0 and 1 (got {p}')
    num_edges = edge_index.size(1)
    edge_mask = edge_index.new_ones(num_edges, dtype=torch.bool)
    if not training or p == 0.0:
        return edge_index, edge_mask, edge_index
    if random_walk is None:
        raise ImportError('`dropout_path` requires `torch-cluster`.')
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_orders = None
    ori_edge_index = edge_index
    if not is_sorted:
        edge_orders = torch.arange(num_edges, device=edge_index.device)
        edge_index, edge_orders = sort_edge_index(edge_index, edge_orders, num_nodes=num_nodes)
    row, col = edge_index
    sample_mask = torch.rand(row.size(0), device=edge_index.device) <= p
    start = row[sample_mask].repeat(walks_per_node)
    deg = degree(row, num_nodes=num_nodes)
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])
    n_id, e_id = random_walk(rowptr, col, start, walk_length, 1.0, 1.0)
    e_id = e_id[e_id != -1].view(-1)  # filter illegal edges
    if edge_orders is not None:
        e_id = edge_orders[e_id]
    edge_mask[e_id] = False
    edge_index_sampled = ori_edge_index[:, edge_mask]
    # 稀疏子图mask：直接返回采样到的边的edge_index
    subgraph_mask = edge_index_sampled
    return edge_index_sampled, edge_mask, subgraph_mask

def n_id_list_to_edge_index1(n_id_list,num_node):
    edge_index=torch.zeros((num_node,num_node))

    for n_id in n_id_list:
        for id1 in torch.unique(n_id):
            for id2 in torch.unique(n_id):
                if id1!=id2:
                    edge_index[id1][id2]=1
    return edge_index
    
def n_id_list_to_edge_index(n_id_list, num_node):
    # 稀疏实现：返回采样到的节点对的edge_index稀疏列表
    edge_indices = []
    for n_id in n_id_list:
        unique_ids = torch.unique(n_id)
        if unique_ids.numel() > 1:
            src = unique_ids.repeat_interleave(unique_ids.numel() - 1)
            dst = unique_ids.repeat(unique_ids.numel() - 1)
            mask = src != dst
            edge_indices.append(torch.stack([src[mask], dst[mask]], dim=0))
    if edge_indices:
        return torch.cat(edge_indices, dim=1)
    else:
        return torch.empty((2, 0), dtype=torch.long, device=n_id_list.device)

