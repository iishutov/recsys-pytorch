from torch_geometric.typing import EdgeType, Metadata, NodeType

import torch
import torch_geometric.nn as gnn
import torch.nn.functional as F

class GNNSAGE(torch.nn.Module):
    def __init__(self, channels: int, dropout_p: float = None):
        super().__init__()
        self.conv1 = gnn.SAGEConv((-1, -1), channels)
        self.dropout_p = dropout_p
        self.conv2 = gnn.SAGEConv((-1, -1), channels)

    def forward(self, x_dict, edge_index):
        h_dict = self.conv1(x_dict, edge_index).relu()
        if self.dropout_p is not None:
            h_dict = F.dropout(h_dict, self.dropout_p, self.training)
        h_dict = self.conv2(h_dict, edge_index)
        
        return h_dict

class IPDecoder(torch.nn.Module):
    def forward(self, x_dict, edge_label_index):
        users_edges, movies_edges = edge_label_index
        x_users = x_dict['user'][users_edges]
        x_movies = x_dict['movie'][movies_edges]

        return (x_users * x_movies).sum(dim=-1)

class MLPDecoder(torch.nn.Module):
    def __init__(self, channels: int, dropout_p: float = None):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * channels, channels)
        self.dropout_p = dropout_p
        self.lin2 = torch.nn.Linear(channels, 1)

    def forward(self, x_dict, edge_label_index):
        users_edges, movies_edges = edge_label_index
        x = torch.cat([x_dict['user'][users_edges], x_dict['movie'][movies_edges]], dim=1)

        h = self.lin1(x).relu()
        if self.dropout_p is not None:
            h = F.dropout(h, self.dropout_p, self.training)
        out = self.lin2(h)

        return out.flatten()

class GNN(torch.nn.Module):
    def __init__(self, metadata: Metadata,
                 hidden_channels: int = 64, decoder: str = 'IP',
                 dropout_encoder_p=None, dropout_nndecoder_p=None):
        '''
        decoder: str
            'IP' - Inner product,
            'NN' - 2-layer neural network.
        '''
        if decoder not in {'IP', 'NN'}:
            raise Exception("Decoder is '{decoder}', which is not a possible option ('IP', 'NN').")

        super().__init__()
        self.encoder = GNNSAGE(hidden_channels, dropout_p=dropout_encoder_p)
        self.encoder = gnn.to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = \
            IPDecoder() if decoder == 'IP' else \
            MLPDecoder(hidden_channels, dropout_nndecoder_p)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        h_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(h_dict, edge_label_index)