import pandas as pd

def get_train_csr_matrix(train_ratio: float = 0.8, watched_pct_threshold: int = 50):
    from scipy.sparse import csr_matrix
    
    df_inter = pd.read_csv('./data/processed/interactions.csv')
    train_size = int(train_ratio * len(df_inter))
    df_train = df_inter[:train_size]

    users_mapping = {user_id: idx for idx, user_id in
                     enumerate(df_train['user_id'].unique())}
    
    movies_mapping = {movie_id: idx for idx, movie_id in
                      enumerate(df_train['item_id'].unique())}

    users_rev_mapping = {idx: user_id for user_id, idx in users_mapping.items()}
    movies_rev_mapping = {idx: movie_id for movie_id, idx in movies_mapping.items()}

    users = df_train['user_id'].map(users_mapping)
    movies = df_train['item_id'].map(movies_mapping)

    is_liked = (df_train['watched_pct'] > watched_pct_threshold).astype(int).to_list()

    csr_mat = csr_matrix(
        (is_liked, (users, movies)),
        shape=(len(users_mapping), len(movies_mapping)))

    return csr_mat, users_rev_mapping, movies_rev_mapping

def get_test_warm_data(train_ratio=0.8, watch_threshold=1):
    df_inter = pd.read_csv('./data/processed/interactions.csv')
    train_size = int(train_ratio * len(df_inter))
    df_test = df_inter[train_size:]
    train_count = df_inter[:train_size]['user_id'].value_counts().to_dict()
    movies = set(df_inter[:train_size]['item_id'].unique())

    test_mask = df_test.apply(lambda inter:
        train_count.get(inter['user_id'], 0) >= watch_threshold and
        inter['item_id'] in movies,
        axis=1)

    df_test = df_test[test_mask]

    return df_test