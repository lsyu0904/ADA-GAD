import scipy.io
import torch
from torch_geometric.data import Data
import numpy as np
import os
import glob

mat_files = glob.glob('data/*.mat')

for mat_path in mat_files:
    dataset_name = os.path.splitext(os.path.basename(mat_path))[0]
    save_dir = f'data/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)

    print(f'正在处理 {mat_path} ...')
    mat = scipy.io.loadmat(mat_path)
    # adj = mat['Network']  # 如果后续只用edge_index，可直接用稀疏格式
    features = mat['Attributes']
    labels = mat['Label']

    # 邻接矩阵转edge_index
    # if hasattr(adj, 'tocoo'):
    #     adj = adj.tocoo()
    #     edge_index = np.vstack((adj.row, adj.col))
    # else:
    edge_index = np.vstack(np.where(mat['Network'] != 0))

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long).squeeze()

    data = Data(x=x, edge_index=edge_index, y=y)
    pt_path = f'{save_dir}/{dataset_name}.pt'
    torch.save(data, pt_path)
    print(f'已保存为 {pt_path}')