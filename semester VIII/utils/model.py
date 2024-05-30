from typing import Dict

import torch
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.loader import NeighborLoader
from torch.nn.modules.loss import _Loss

class MaxMarginLoss(_Loss):
    def __init__(self, margin: float, alpha: float,
                 model: torch.nn.Module, device: torch.device):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.model = model
        self.device = device

    def forward(self, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(len(positives), device=self.device)

        reg = 0
        if self.alpha != None:
            for param in self.model.parameters():
                reg += param.pow(2).sum()
            reg = self.alpha * reg / len(positives)

        return torch.max(z, -positives + negatives + self.margin).mean() + reg

class _GNNSAGE(torch.nn.Module):
    def __init__(self, num_layers: int, user_dim: int, movie_dim: int, channels: int,
                 normalize: bool = False, aggr_type: str = 'mean',
                 dropout_p: float = None):
        super().__init__()
        self.dropout_p = dropout_p
        self.convs = torch.nn.ModuleList()

        kwargs = dict(
            out_channels=channels, normalize=normalize,
            aggr = 'max' if aggr_type == 'pool-max' else 'mean'
        )
        if aggr_type == 'pool-max':
            kwargs.update(dict(project=True))

        for layer_num in range(num_layers):
            in_channels = (user_dim, movie_dim) if layer_num == 0 \
                            else (channels, channels)
            conv = gnn.HeteroConv({
                ('user', 'watched', 'movie'):
                    gnn.SAGEConv(in_channels=in_channels, **kwargs),
                ('movie', 'rev_watched', 'user'):
                    gnn.SAGEConv(in_channels=in_channels[::-1], **kwargs),
            }, aggr='sum')
            self.convs.append(conv)

    def forward(self,
                x_dict: Dict[NodeType, torch.Tensor],
                edge_index_dict: Dict[EdgeType, torch.Tensor]) \
                    -> Dict[NodeType, torch.Tensor]:
        
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i != len(self.convs)-1:
                x_dict = {node_type: F.leaky_relu(x)
                          for node_type, x in x_dict.items()}
                if self.dropout_p is not None:
                    x_dict = {node_type: F.dropout(x, self.dropout_p, self.training)
                              for node_type, x in x_dict.items()}
        return x_dict

class _IPDecoder(torch.nn.Module):
    def forward(self,
                h_dict: Dict[NodeType, torch.Tensor],
                edge_label_index: torch.Tensor) \
                    -> torch.Tensor:
        
        users_idx, movies_idx = edge_label_index
        h_users = h_dict['user'][users_idx]
        h_movies = h_dict['movie'][movies_idx]

        return (h_users * h_movies).sum(dim=-1)

class GNN(torch.nn.Module):
    def __init__(self, num_layers: int, user_dim: int, movie_dim: int,
                 hidden_channels: int = 64, normalize: bool = False,
                 aggr_type: str = 'mean', dropout_p=None):
        super().__init__()

        if aggr_type not in {'mean', 'max-pool'}:
            raise Exception("aggr_type for SAGEConv layer must be 'mean' or 'max-pool'")

        self.encoder = _GNNSAGE(num_layers, user_dim, movie_dim, hidden_channels,
                               normalize, aggr_type, dropout_p)
        self.decoder = _IPDecoder()

    def forward(self,
                x_dict: Dict[NodeType, torch.Tensor],
                edge_index_dict: Dict[EdgeType, torch.Tensor],
                edge_label_index: torch.Tensor) \
                    -> torch.Tensor:
        
        h_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(h_dict, edge_label_index)
    
    def get_movies_embeddings(self, movie_loader: NeighborLoader, device: torch.device) \
            -> list:
        movie_embs = []
        for batch in movie_loader:
            batch = batch.to(device)
            batch_size = batch['movie'].batch_size
            batch_movie_embs = self.encoder(batch.x_dict, batch.edge_index_dict)\
                ['movie'][:batch_size]
            movie_embs.append(batch_movie_embs)

        return movie_embs
    
    def recommend(self, users_idx: torch.Tensor, data: HeteroData,
                  top_count: int, k: int,
                  model_device: torch.device, faiss_device: torch.device) \
                    -> torch.Tensor:
        
        from torch_geometric import EdgeIndex
        from torch_geometric.nn import MIPSKNNIndex

        num_users, num_movies = data['user'].num_nodes, data['movie'].num_nodes
        edges = EdgeIndex(data['user', 'movie'].edge_index.contiguous().to(model_device),
                          sparse_size=(num_users, num_movies))
        time = data['user', 'movie'].time

        loader_kwargs = dict(
            data=data, batch_size=1024,
            num_neighbors=[5, 5, 5],
            time_attr='time', temporal_strategy='last',
            num_workers=4)

        movie_loader = NeighborLoader(
            input_nodes='movie',
            input_time=(time[-1]).repeat(num_movies),
            **loader_kwargs)

        movie_embs = self.get_movies_embeddings(movie_loader, model_device)
        movie_embs = torch.cat(movie_embs, dim=0).to(faiss_device)
        if top_count is not None: movie_embs = movie_embs[:top_count]
        
        mipsknn = MIPSKNNIndex(movie_embs)
        
        user_neighbors_loader = NeighborLoader(
            input_nodes=('user', users_idx),
            input_time=(time[-1]).repeat(num_users),
            **loader_kwargs)
        
        recs = torch.empty(size=(len(users_idx), k))
        users_infered = 0
        for batch in user_neighbors_loader:
            batch = batch.to(model_device)
            batch_size = batch['user'].batch_size
            user_embs = self.encoder(batch.x_dict, batch.edge_index_dict)\
                ['user'][:batch_size].reshape(batch_size, -1).to(faiss_device)

            batch_users_idx = batch['user'].n_id[:batch_size]
            users_edges = edges[:, (edges[0] == batch_users_idx.unsqueeze(1)).\
                                sum(0).bool().squeeze()].to(faiss_device)
            
            if top_count is not None:
                users_edges = users_edges[:, users_edges[1] < top_count]

            user_mapping = {input_id: idx for idx, input_id in
                            enumerate(batch_users_idx.tolist())}
            users_edges[0, :].apply_(user_mapping.get)

            batch_recs = mipsknn.search(user_embs, k, exclude_links=users_edges)[1]
            recs[users_infered:users_infered+batch_size] = batch_recs.clone().detach()
            users_infered += batch_size

        return recs