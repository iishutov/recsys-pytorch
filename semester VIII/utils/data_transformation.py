import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric import EdgeIndex

def data_to_heterograph(data_path: str) -> HeteroData:
    from torch_geometric.transforms import ToUndirected

    df_inter = pd.read_csv('./data/processed/interactions.csv')
    df_items = pd.read_csv('./data/processed/items.csv')
    df_users = pd.read_csv('./data/processed/users.csv')

    with np.load(data_path) as file:
        X_users, X_movies = file['X_users'], file['X_movies']

    users_mapping = {user_id: idx \
        for idx, user_id in df_users['user_id'].items()}
    
    movies_mapping = {item_id: idx \
        for idx, item_id in df_items['item_id'].items()}
    
    users_rev_mapping = {idx: user_id for user_id, idx in users_mapping.items()}
    movies_rev_mapping = {idx: movie_id for movie_id, idx in movies_mapping.items()}

    edges = np.vstack([
        df_inter['user_id'].map(users_mapping).values,
        df_inter['item_id'].map(movies_mapping).values
    ])

    time = pd.to_datetime(df_inter['last_watch_dt'], format='%Y-%m-%d').\
        values.astype(np.int64) // 10**9

    data = HeteroData()
    data['movie'].x = torch.tensor(X_movies)
    data['user'].x = torch.tensor(X_users)
    
    data['movie'].num_features = X_movies.shape[1]
    data['user'].num_features = X_users.shape[1]

    data['user', 'watched', 'movie'].edge_index = torch.tensor(edges)
    data['user', 'watched', 'movie'].time = torch.tensor(time)
    
    return ToUndirected()(data), users_rev_mapping, movies_rev_mapping

def sparse_batch_narrow(edges: EdgeIndex, users_infered: int, batch_size: int)\
        -> EdgeIndex:
    batch_edges = edges.sparse_narrow(0, users_infered, batch_size).cpu()

    mapping = {int(node_idx): batch_node_idx for batch_node_idx, node_idx \
                    in enumerate(batch_edges[0].unique())}
    
    batch_edges[0].apply_(mapping.get)
    batch_edges._sparse_size = (len(mapping), batch_edges._sparse_size[1])

    return batch_edges