from typing import Dict

import torch
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.loader import NeighborLoader

class GNNSAGE(torch.nn.Module):
    def __init__(self, num_layers: int, channels: int, dropout_p: float = None):
        super().__init__()
        self.dropout_p = dropout_p
        self.convs = torch.nn.ModuleList()

        conv1 = gnn.HeteroConv({
            ('user', 'watched', 'movie'): gnn.SAGEConv((-1, -1), channels),
            ('movie', 'rev_watched', 'user'): gnn.SAGEConv((-1, -1), channels),
            }, aggr='sum')
        self.convs.append(conv1)

        for _ in range(num_layers-1):
            sage_conv = gnn.SAGEConv((-1, -1), channels)
            conv = gnn.HeteroConv({
                ('user', 'watched', 'movie'): sage_conv,
                ('movie', 'rev_watched', 'user'): sage_conv,
            }, aggr='sum')
            self.convs.append(conv)

    def forward(self,
                x_dict: Dict[NodeType, torch.Tensor],
                edge_index_dict: Dict[EdgeType, torch.Tensor]) \
                    -> Dict[NodeType, torch.Tensor]:
        
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {node_type: F.leaky_relu(x) for node_type, x in x_dict.items()}
            if (self.dropout_p is not None) and (i != len(self.convs)):
                x_dict = F.dropout(x_dict, self.dropout_p, self.training)
        
        return x_dict

class IPDecoder(torch.nn.Module):
    def forward(self,
                x_dict: Dict[NodeType, torch.Tensor],
                edge_label_index: torch.Tensor) \
                    -> torch.Tensor:
        
        users_idx, movies_idx = edge_label_index
        x_users = x_dict['user'][users_idx]
        x_movies = x_dict['movie'][movies_idx]

        return (x_users * x_movies).sum(dim=-1)

class GNN(torch.nn.Module):
    def __init__(self, num_layers: int = 3, hidden_channels: int = 64,
                 dropout_encoder_p=None):
        super().__init__()

        self.encoder = GNNSAGE(num_layers, hidden_channels, dropout_encoder_p)
        #self.encoder = gnn.to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = IPDecoder()

    def forward(self,
                x_dict: Dict[NodeType, torch.Tensor],
                edge_index_dict: Dict[EdgeType, torch.Tensor],
                edge_label_index: torch.Tensor) \
                    -> torch.Tensor:
        
        h_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(h_dict, edge_label_index)
    
    def get_movies_embeddings(self, movie_loader: NeighborLoader, device: torch.device):
        movie_embs = []
        for batch in movie_loader:
            batch = batch.to(device)
            batch_size = batch['movie'].batch_size
            batch_movie_embs = self.encoder(batch.x_dict, batch.edge_index_dict)\
                ['movie'][:batch_size]
            movie_embs.append(batch_movie_embs)

        return movie_embs