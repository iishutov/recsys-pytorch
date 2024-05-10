import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric import EdgeIndex

def data_to_heterograph(data_path: str) -> HeteroData:
    from torch_geometric.transforms import ToUndirected

    df_inter = pd.read_csv('./processed_data/interactions.csv')
    df_items = pd.read_csv('./processed_data/items.csv')
    df_users = pd.read_csv('./processed_data/users.csv')

    with np.load(data_path) as file:
        X_users, X_movies = file['X_users'], file['X_movies']

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
    data['movie'].x = torch.tensor(X_movies)
    data['user'].x = torch.tensor(X_users)

    data['user', 'watched', 'movie'].edge_index = torch.tensor(edges)
    data['user', 'watched', 'movie'].time = torch.tensor(time)
    
    return ToUndirected()(data)

def get_train_csr_matrix(train_ratio: float = 0.8):
    from scipy.sparse import csr_matrix
    
    df_inter = pd.read_csv('./processed_data/interactions.csv')
    train_size = int(train_ratio * len(df_inter))
    df_inter_train = df_inter[:train_size]

    users_mapping = {user_id: idx for idx, user_id in
                     enumerate(set(df_inter_train['user_id']))}
    
    movies_mapping = {movie_id: idx for idx, movie_id in
                      enumerate(set(df_inter_train['item_id']))}

    users_rev_mapping = {idx: user_id for user_id, idx in users_mapping.items()}
    movies_rev_mapping = {idx: movie_id for movie_id, idx in movies_mapping.items()}

    users = df_inter_train['user_id'].map(users_mapping)
    movies = df_inter_train['item_id'].map(movies_mapping)

    csr_mat = csr_matrix(
        (np.ones(len(df_inter_train)), (users, movies)),
        shape=(len(users_mapping), len(movies_mapping)))

    return csr_mat, users_rev_mapping, movies_rev_mapping

def get_test_data(train_ratio=0.8, watch_threshold = 1):
    df_inter = pd.read_csv('./processed_data/interactions.csv')
    train_size = int(train_ratio * len(df_inter))
    df_inter_test = df_inter[train_size:]
    train_count = df_inter[:train_size]['user_id'].value_counts().to_dict()
    watch_threshold = 5
    movies_mapping = {movie_id: idx for idx, movie_id in
                      enumerate(set(df_inter[:train_size]['item_id']))}

    test_mask = df_inter_test.apply(lambda view:
                                train_count.get(view['user_id'], 0) >= watch_threshold and
                                view['item_id'] in set(movies_mapping.keys()),
                                axis=1)

    df_inter_test = df_inter_test[test_mask]

    return df_inter_test

def compute_metrics(users_idx: np.ndarray, test_interactions: dict, recs: np.ndarray):
    k = recs.shape[1]
    precision = 0.
    recall = 0.
    
    for i, user_idx in enumerate(users_idx):
        interactions = test_interactions[user_idx]
        relevant_count = len(interactions.intersection(recs[i]))

        precision += relevant_count / k
        recall += relevant_count / len(interactions)

    precision /= len(users_idx)
    recall /= len(users_idx)

    return precision, recall

def sparse_batch_narrow(edges: EdgeIndex, users_infered: int, batch_size: int):
    batch_edges = edges.sparse_narrow(0, users_infered, batch_size).cpu()

    mapping = {int(node_idx): batch_node_idx for batch_node_idx, node_idx \
                    in enumerate(batch_edges[0].unique())}
    
    batch_edges[0].apply_(mapping.get)
    batch_edges._sparse_size = (len(mapping), batch_edges._sparse_size[1])

    return batch_edges