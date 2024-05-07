from typing import Dict, List

import torch
import torch_geometric.nn as gnn
from torch_geometric import EdgeType, NodeType
from torch_geometric.typing import EdgeType

from tqdm import tqdm

default_metapath = [
    ('user', 'watched', 'movie'),
    ('movie', 'rev_watched', 'user'),
]

class Metapath2Vec:
    def __init__(self,
                 edge_index_dict: Dict[EdgeType, torch.Tensor],
                 num_nodes_dict: Dict[NodeType, int],
                 embedding_dim: int = 128,
                 metapath: List[EdgeType] = default_metapath,
                 walks_per_node: int = 5,
                 num_negative_samples: int = 5,
                 walk_length: int = 20,
                 context_size: int = 5,
                 sparse: bool = True,
                 batch_size: int = 256,
                 shuffle: bool = True,
                 learning_rate: float = 0.01,
                 device: str = 'cpu'):
        
        self.mp2v = gnn.MetaPath2Vec(edge_index_dict, embedding_dim, metapath,
            walk_length, context_size, walks_per_node, num_negative_samples,
            num_nodes_dict, sparse).to(device)
        
        self.loader = self.mp2v.loader(batch_size=batch_size, shuffle=shuffle)
        self.optimizer = torch.optim.SparseAdam(list(self.mp2v.parameters()), lr=learning_rate)

    def __train_epoch__(self):
        self.mp2v.train()
        for pos_rw, neg_rw in tqdm(self.loader):
            self.optimizer.zero_grad()
            loss = self.mp2v.loss(pos_rw.device(), neg_rw.device())
            loss.backward()
            self.optimizer.step()
                
        return loss

    def train(self, epochs: int = 8, verbose: bool = True):
        for epoch in range(1, epochs+1):
            loss = self.train_epoch()
            if verbose:
                print((f'Epoch: {epoch}, Loss: {loss :.4f}'))

    def get_embeddings(self, node_type: str):
        return self.mp2v.forward(node_type).detach()