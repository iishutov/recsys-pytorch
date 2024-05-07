import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

def data_to_heterograph(data_path: str):

    import pandas as pd
    import numpy as np

    df_inter = pd.read_csv('./mod_data/interactions.csv')
    df_items = pd.read_csv('./mod_data/items.csv')
    df_users = pd.read_csv('./mod_data/users.csv')

    with np.load(data_path) as file:
        X_users, X_items = file['X_users'], file['X_items']


    user_id_to_idx = {user_id: idx \
        for idx, user_id in df_users['user_id'].items()}
    
    item_id_to_idx = {item_id: idx \
        for idx, item_id in df_items['item_id'].items()}

    edges = np.vstack([
        df_inter['user_id'].map(user_id_to_idx).values,
        df_inter['item_id'].map(item_id_to_idx).values
    ])

    time = pd.to_datetime(df_inter['last_watch_dt'], format='%Y-%m-%d').\
        values.astype(np.int64) // 10**9

    data = HeteroData()
    data['movie'].x = torch.Tensor(X_items)
    data['user'].x = torch.Tensor(X_users)
    data['user', 'watched', 'movie'].edge_index = torch.tensor(edges)
    data['user', 'watched', 'movie'].time = torch.tensor(time)
    
    return T.ToUndirected()(data)