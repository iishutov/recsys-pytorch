import torch
from torch_geometric.data import HeteroData
from torch_geometric import EdgeIndex

def data_to_heterograph(data_path: str, temporal_order: bool = False) -> HeteroData:
    from pandas import read_csv, to_datetime
    from numpy import load, vstack, int64
    from torch_geometric.transforms import ToUndirected

    df_inter = read_csv('./mod_data/interactions.csv')
    df_items = read_csv('./mod_data/items.csv')
    df_users = read_csv('./mod_data/users.csv')

    with load(data_path) as file:
        X_users, X_items = file['X_users'], file['X_items']

    user_id_to_idx = {user_id: idx \
        for idx, user_id in df_users['user_id'].items()}
    
    item_id_to_idx = {item_id: idx \
        for idx, item_id in df_items['item_id'].items()}

    edges = vstack([
        df_inter['user_id'].map(user_id_to_idx).values,
        df_inter['item_id'].map(item_id_to_idx).values
    ])

    time = to_datetime(df_inter['last_watch_dt'], format='%Y-%m-%d').\
        values.astype(int64) // 10**9

    data = HeteroData()
    data['movie'].x = torch.tensor(X_items)
    data['user'].x = torch.tensor(X_users)

    if temporal_order:
        sorted_idx = time.argsort()
        time = time[sorted_idx]
        edges = edges[:, sorted_idx]

    data['user', 'watched', 'movie'].edge_index = torch.tensor(edges)
    data['user', 'watched', 'movie'].time = torch.tensor(time)
    
    return ToUndirected()(data)

def sparse_batch_narrow(edges: EdgeIndex, users_infered: int, batch_size: int):
    batch_edges = edges.sparse_narrow(0, users_infered, batch_size)

    mapping = {int(node_idx): batch_idx for batch_idx, node_idx \
                    in enumerate(batch_edges[0].unique())}
    
    batch_edges[0].apply_(mapping.get)
    batch_edges._sparse_size = (len(mapping), batch_edges._sparse_size[1])

    return batch_edges